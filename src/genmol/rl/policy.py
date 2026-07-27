import itertools
import os
import random
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

import safe as sf
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
    conditions: list[str] | None = None
    raw_smiles: list[str | None] | None = None


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
        self._motif_token_cache = {}

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

    def encode_motif_fragment(self, fragment):
        cached = self._motif_token_cache.get(fragment)
        if cached is not None:
            return cached.clone()
        # Official GenMol fragment_completion uses standard SAFE encoding for
        # both GenMol variants; the v2 tokenizer contains these tokens too.
        encoded_fragment = (
            sf.SAFEConverter(ignore_stereo=True).encoder(
                fragment,
                allow_empty=True,
            )
            + '.'
        )
        token_ids = self.model.tokenizer(
            [encoded_fragment],
            return_tensors='pt',
            truncation=False,
        )['input_ids'][0].detach().cpu()
        max_positions = int(self.model.config.model.max_position_embeddings)
        if token_ids.numel() > max_positions:
            raise ValueError(
                'motif prompt exceeds model maximum context: '
                f'{token_ids.numel()} vs {max_positions} for {fragment!r}'
            )
        if token_ids.numel() < 2:
            raise ValueError(
                f'motif prompt must contain BOS and EOS: {fragment!r}'
            )
        if int(token_ids[0]) != self.bos_index or int(token_ids[-1]) != self.eos_index:
            raise ValueError(
                'motif tokenizer output must start with BOS and end with EOS: '
                f'{fragment!r}'
            )
        self._motif_token_cache[fragment] = token_ids
        return token_ids.clone()

    def motif_base_sequence_length(self, fragment):
        return int(self.encode_motif_fragment(fragment).numel())

    def rollout_motif_extension(
        self,
        specs,
        fragments,
        generation_batch_size,
        seed,
        gamma=0.3,
        guidance_weight=2.0,
    ):
        if not specs:
            raise ValueError('specs must be non-empty')
        if len(fragments) != len(specs):
            raise ValueError(
                f'fragments must match specs length: {len(fragments)} vs {len(specs)}'
            )
        if generation_batch_size <= 0:
            raise ValueError('generation_batch_size must be positive')
        gamma = float(gamma)
        guidance_weight = float(guidance_weight)
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f'gamma must be in [0, 1], got {gamma}')
        if guidance_weight <= 0.0:
            raise ValueError(
                f'guidance_weight must be positive, got {guidance_weight}'
            )

        torch.manual_seed(seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)

        base_rows = [self.encode_motif_fragment(fragment) for fragment in fragments]
        row_lengths = [
            int(base.numel()) + int(spec.add_seq_len)
            for base, spec in zip(base_rows, specs)
        ]
        max_positions = int(self.model.config.model.max_position_embeddings)
        overlength = [
            (idx, length)
            for idx, length in enumerate(row_lengths)
            if length > max_positions
        ]
        if overlength:
            idx, length = overlength[0]
            raise ValueError(
                'motif-extension prompt plus generated region exceeds model context: '
                f'{length} vs {max_positions} at sample {idx}'
            )
        global_max_length = max(row_lengths)
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
                            candidate.generation_temperature
                            != ref_spec.generation_temperature
                            or candidate.randomness != ref_spec.randomness
                        ):
                            break
                        end += 1
                    chunk_specs = specs[start:end]
                    chunk_bases = base_rows[start:end]
                    token_ids = torch.full(
                        (len(chunk_specs), global_max_length),
                        fill_value=self.pad_index,
                        device=self.device,
                        dtype=torch.long,
                    )
                    completion_mask = torch.zeros_like(
                        token_ids,
                        dtype=torch.bool,
                    )
                    context_mask = torch.zeros_like(
                        token_ids,
                        dtype=torch.bool,
                    )
                    for row_idx, (base, spec) in enumerate(
                        zip(chunk_bases, chunk_specs)
                    ):
                        base = base.to(self.device)
                        base_length = int(base.numel())
                        add_length = int(spec.add_seq_len)
                        token_ids[row_idx, :base_length - 1] = base[:-1]
                        mask_start = base_length - 1
                        mask_end = mask_start + add_length
                        token_ids[row_idx, mask_start:mask_end] = self.mask_index
                        token_ids[row_idx, mask_end] = base[-1]
                        completion_mask[row_idx, mask_start:mask_end] = True
                        context_mask[row_idx, 1:base_length - 1] = True

                    x = token_ids
                    num_steps = max(
                        self.model.mdlm.get_num_steps_confidence(x),
                        2,
                    )
                    for step_idx in range(num_steps):
                        logits = self.forward_logits(x)
                        if gamma > 0.0 and guidance_weight != 1.0:
                            poor_x = x.clone()
                            for row_idx in range(poor_x.size(0)):
                                context_positions = torch.nonzero(
                                    context_mask[row_idx],
                                    as_tuple=False,
                                ).view(-1).tolist()
                                num_mask = int(len(context_positions) * gamma)
                                if num_mask > 0:
                                    selected = random.sample(
                                        context_positions,
                                        num_mask,
                                    )
                                    poor_x[row_idx, selected] = self.mask_index
                            poor_logits = self.forward_logits(poor_x)
                            logits = (
                                guidance_weight * logits
                                + (1.0 - guidance_weight) * poor_logits
                            )
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
        completion_mask = torch.cat(chunk_masks, dim=0)
        safe_strings = self._decode_safe_strings(token_ids)
        raw_smiles = self._decode_smiles(safe_strings)
        return RolloutBatch(
            prompt_ids=token_ids[:, :0].detach().clone(),
            completion_ids=token_ids.detach().clone(),
            completion_mask=completion_mask,
            full_token_ids=token_ids,
            specs=list(specs),
            safe_strings=safe_strings,
            smiles=list(raw_smiles),
            conditions=list(fragments),
            raw_smiles=list(raw_smiles),
        )

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
