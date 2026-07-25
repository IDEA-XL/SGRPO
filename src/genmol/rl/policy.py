import itertools
import os
import random
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

import torch
from torch.nn.parallel import DistributedDataParallel

from genmol.mm.checkpoint import load_checkpoint_payload, require_unimodal_checkpoint, stamp_checkpoint_variant, UNIMODAL_VARIANT
from genmol.model import GenMol
from genmol.rl.cpgrpo import get_per_token_logps
from genmol.utils.bracket_safe_converter import bracketsafe2safe
from genmol.utils.utils_chem import safe_to_smiles


@dataclass
class RolloutBatch:
    prompt_ids: torch.Tensor
    completion_ids: torch.Tensor
    completion_mask: torch.Tensor
    full_token_ids: torch.Tensor
    specs: list
    safe_strings: list[str]
    smiles: list[str | None]


def _move_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _move_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_cpu(item) for item in value)
    return value


class GenMolCpGRPOPolicy:
    def __init__(self, checkpoint_path, device, bf16=True, trainable=True):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.bf16 = bf16
        checkpoint = load_checkpoint_payload(checkpoint_path)
        require_unimodal_checkpoint(checkpoint, checkpoint_path)
        self.model = GenMol.load_from_checkpoint(checkpoint_path, map_location='cpu')
        self.model.to(self.device)

        if self.model.ema:
            self.model.ema.move_shadow_params_to_device(self.device)
            self.model.ema.copy_to(itertools.chain(self.model.backbone.parameters()))
        self.mask_index = self.model.mask_index
        self.bos_index = self.model.bos_index
        self.eos_index = self.model.eos_index
        self.pad_index = self.model.tokenizer.pad_token_id
        self.use_bracket_safe = bool(self.model.config.training.get('use_bracket_safe'))

        if not trainable:
            self.freeze()

    @property
    def backbone(self):
        return self.model.backbone

    def enable_gradient_checkpointing(self, gradient_checkpointing_kwargs=None):
        kwargs = gradient_checkpointing_kwargs or {}
        if hasattr(self._unwrap_backbone(), 'gradient_checkpointing_enable'):
            self._unwrap_backbone().gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)

    def freeze(self):
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def train(self):
        self.model.train()

    @property
    def autocast_context(self):
        if self.device.type != 'cuda':
            return nullcontext()
        if not self.bf16:
            return nullcontext()
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)

    def _unwrap_backbone(self):
        backbone = self.model.backbone
        if isinstance(backbone, DistributedDataParallel):
            backbone = backbone.module
        while hasattr(backbone, 'module'):
            backbone = backbone.module
        return backbone

    def trainable_parameters(self):
        return self._unwrap_backbone().parameters()

    def update_ema(self):
        if self.model.ema:
            self.model.ema.update(itertools.chain(self._unwrap_backbone().parameters()))

    def sync_from(self, other_policy, alpha):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'alpha must be in [0, 1], got {alpha}')

        source_state = other_policy._unwrap_backbone().state_dict()
        target_backbone = self._unwrap_backbone()
        target_state = target_backbone.state_dict()

        mixed_state = {}
        for key, target_value in target_state.items():
            source_value = source_state[key].detach().to(device=target_value.device, dtype=target_value.dtype)
            target_value = target_value.detach()
            if torch.is_floating_point(target_value):
                mixed_state[key] = target_value.mul(1.0 - alpha).add(source_value, alpha=alpha)
            else:
                mixed_state[key] = source_value
        target_backbone.load_state_dict(mixed_state, strict=True)

    @contextmanager
    def backbone_eval_mode(self):
        backbone = self.model.backbone
        was_training = backbone.training
        backbone.eval()
        try:
            yield
        finally:
            backbone.train(was_training)

    def forward_logits(self, input_ids):
        input_ids = input_ids.clone()
        attention_mask = input_ids != self.pad_index
        batch_size, seq_len = input_ids.shape
        max_position_embeddings = int(self.model.config.model.max_position_embeddings)
        if seq_len > max_position_embeddings:
            raise ValueError(
                'Input sequence length exceeds model maximum context: '
                f'{seq_len} vs {max_position_embeddings}'
            )
        token_type_ids = torch.zeros((batch_size, seq_len), device=input_ids.device, dtype=torch.long)
        position_ids = torch.arange(seq_len, device=input_ids.device, dtype=torch.long).unsqueeze(0)
        with self.autocast_context:
            logits = self.model.backbone(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
            )['logits']
        return logits.float()

    def per_token_logps(
        self,
        input_ids,
        logits_to_keep,
        completion_mask,
        mask_seeds,
        gradient_accumulation_steps,
        requires_grad,
        return_normalized_entropy=False,
    ):
        def score_fn(batch):
            return self.forward_logits(batch)

        return get_per_token_logps(
            score_fn=score_fn,
            input_ids=input_ids,
            logits_to_keep=logits_to_keep,
            completion_mask=completion_mask,
            mask_token_id=self.mask_index,
            mask_seeds=mask_seeds,
            gradient_accumulation_steps=gradient_accumulation_steps,
            requires_grad=requires_grad,
            return_normalized_entropy=return_normalized_entropy,
        )

    def _decode_safe_strings(self, token_ids):
        return self.model.tokenizer.batch_decode(token_ids, skip_special_tokens=True)

    def _decode_smiles(self, safe_strings):
        smiles_list = []
        for safe_string in safe_strings:
            try:
                if self.use_bracket_safe:
                    smiles = safe_to_smiles(bracketsafe2safe(safe_string), fix=True)
                else:
                    smiles = safe_to_smiles(safe_string, fix=True)
            except Exception:
                smiles = None

            if smiles:
                smiles = sorted(smiles.split('.'), key=len)[-1]
            smiles_list.append(smiles)
        return smiles_list

    def rollout_specs(self, specs, generation_batch_size, seed):
        if not specs:
            raise ValueError('specs must be non-empty')
        if generation_batch_size <= 0:
            raise ValueError('generation_batch_size must be positive')

        torch.manual_seed(seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)

        global_max_add_len = max(spec.add_seq_len for spec in specs)
        chunk_outputs = []
        chunk_masks = []

        with self.backbone_eval_mode():
            with torch.no_grad():
                start = 0
                while start < len(specs):
                    max_end = min(start + generation_batch_size, len(specs))
                    ref_spec = specs[start]
                    end = start + 1
                    while end < max_end:
                        candidate = specs[end]
                        if (
                            candidate.generation_temperature != ref_spec.generation_temperature
                            or candidate.randomness != ref_spec.randomness
                        ):
                            break
                        end += 1
                    chunk_specs = specs[start:end]
                    chunk_size = len(chunk_specs)
                    token_ids = torch.full(
                        (chunk_size, global_max_add_len + 2),
                        fill_value=self.pad_index,
                        device=self.device,
                        dtype=torch.long,
                    )
                    completion_mask = torch.zeros_like(token_ids, dtype=torch.bool)

                    for row_idx, spec in enumerate(chunk_specs):
                        token_ids[row_idx, 0] = self.bos_index
                        token_ids[row_idx, spec.add_seq_len + 1] = self.eos_index
                        token_ids[row_idx, 1:spec.add_seq_len + 1] = self.mask_index
                        completion_mask[row_idx, 1:spec.add_seq_len + 1] = True

                    x = token_ids
                    num_steps = max(self.model.mdlm.get_num_steps_confidence(x), 2)
                    for step_idx in range(num_steps):
                        logits = self.forward_logits(x)
                        x = self.model.mdlm.step_confidence(
                            logits,
                            x,
                            step_idx,
                            num_steps,
                            chunk_specs[0].generation_temperature,
                            chunk_specs[0].randomness,
                        )

                    chunk_outputs.append(x.detach().clone())
                    chunk_masks.append(completion_mask.detach().clone())
                    start = end

        token_ids = torch.cat(chunk_outputs, dim=0)
        completion_mask = torch.cat(chunk_masks, dim=0)[:, 1:]
        prompt_ids = token_ids[:, :1].detach().clone()
        completion_ids = token_ids[:, 1:].detach().clone()
        safe_strings = self._decode_safe_strings(token_ids)
        smiles = self._decode_smiles(safe_strings)
        return RolloutBatch(
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            full_token_ids=token_ids,
            specs=list(specs),
            safe_strings=safe_strings,
            smiles=smiles,
        )

    def load_ema_state(self, ema_state):
        if self.model.ema and ema_state is not None:
            self.model.ema.load_state_dict(ema_state)
            self.model.ema.move_shadow_params_to_device(self.device)

    def load_backbone_state_dict(self, state_dict):
        self._unwrap_backbone().load_state_dict(state_dict, strict=True)

    def get_backbone_state_dict(self):
        return _move_to_cpu(self._unwrap_backbone().state_dict())

    def save_checkpoint(self, path, step, accelerator=None):
        checkpoint = load_checkpoint_payload(self.checkpoint_path)
        require_unimodal_checkpoint(checkpoint, self.checkpoint_path)
        if accelerator is None:
            backbone_state = _move_to_cpu(self._unwrap_backbone().state_dict())
        else:
            backbone_state = _move_to_cpu(accelerator.get_state_dict(self.model.backbone))

        model_state = _move_to_cpu(self.model.state_dict())
        checkpoint['state_dict'] = {}
        for key, value in model_state.items():
            if key.startswith('backbone.'):
                continue
            checkpoint['state_dict'][key] = value
        checkpoint['state_dict'].update({f'backbone.{key}': value for key, value in backbone_state.items()})
        checkpoint['global_step'] = int(step)
        checkpoint['epoch'] = 0
        checkpoint['optimizer_states'] = []
        checkpoint['lr_schedulers'] = []

        if self.model.ema:
            checkpoint['ema'] = _move_to_cpu(self.model.ema.state_dict())
        stamp_checkpoint_variant(checkpoint, UNIMODAL_VARIANT)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
