import itertools
import os
import random
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

import torch
from torch.nn.parallel import DistributedDataParallel

from genmol.mm.checkpoint import load_checkpoint_payload, require_multimodal_checkpoint
from genmol.mm.model import PocketPrefixGenMol
from genmol.mm.prefix import pad_prefix_embeddings
from genmol.rl.cpgrpo import forward_process, selective_log_softmax
from genmol.utils.bracket_safe_converter import bracketsafe2safe
from genmol.utils.utils_chem import safe_to_smiles


@dataclass
class PocketPrefixRolloutBatch:
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


def _unwrap_module(module):
    if isinstance(module, DistributedDataParallel):
        module = module.module
    while hasattr(module, 'module'):
        module = module.module
    return module


def get_per_token_logps_with_pocket(
    score_fn,
    input_ids,
    logits_to_keep,
    completion_mask,
    mask_token_id,
    mask_seeds,
    gradient_accumulation_steps,
    requires_grad,
    pocket_raw_embeddings,
    pocket_mask,
    return_normalized_entropy=False,
):
    if input_ids.dim() != 3:
        raise ValueError(f'Expected input_ids to have 3 dimensions, got {input_ids.dim()}')
    if completion_mask.dim() != 2:
        raise ValueError(f'Expected completion_mask to have 2 dimensions, got {completion_mask.dim()}')
    if pocket_raw_embeddings.dim() != 3:
        raise ValueError('pocket_raw_embeddings must have shape [batch, max_prefix_len, esm_dim]')
    if pocket_mask.dim() != 2:
        raise ValueError('pocket_mask must have shape [batch, max_prefix_len]')

    num_iterations, batch_size, seq_len = input_ids.size()
    if logits_to_keep <= 0 or logits_to_keep > seq_len:
        raise ValueError(f'logits_to_keep must be in [1, {seq_len}], got {logits_to_keep}')
    if completion_mask.size(0) != batch_size or completion_mask.size(1) != logits_to_keep:
        raise ValueError(
            'completion_mask must have shape '
            f'[{batch_size}, {logits_to_keep}], got {list(completion_mask.shape)}'
        )
    if len(mask_seeds) != num_iterations:
        raise ValueError(f'Expected {num_iterations} mask seeds, got {len(mask_seeds)}')
    if pocket_raw_embeddings.size(0) != batch_size or pocket_mask.size(0) != batch_size:
        raise ValueError('pocket tensors must share the same batch size as input_ids')

    prompt_length = seq_len - logits_to_keep
    full_completion_mask = torch.zeros((batch_size, seq_len), device=input_ids.device, dtype=torch.bool)
    full_completion_mask[:, prompt_length:] = completion_mask

    grad_context = nullcontext() if requires_grad else torch.no_grad()
    with grad_context:
        all_perturbed = []
        all_weights = []
        all_expanded_inputs = []
        all_partial_masks = []
        all_pocket_embeddings = []
        all_pocket_masks = []

        for iteration_idx, mask_seed in enumerate(mask_seeds):
            expanded_input = input_ids[iteration_idx]
            perturbed, weights, partial_mask = forward_process(
                batch=expanded_input,
                completion_mask=full_completion_mask,
                mask_id=mask_token_id,
                seed=mask_seed,
                gradient_accumulation_steps=gradient_accumulation_steps,
                accumulate=num_iterations > 1,
            )
            all_perturbed.extend(perturbed)
            all_weights.extend(weights)
            all_expanded_inputs.append(expanded_input)
            all_partial_masks.append(partial_mask)
            for _ in range(3):
                all_pocket_embeddings.append(pocket_raw_embeddings)
                all_pocket_masks.append(pocket_mask)

        perturbed_seq = torch.cat(all_perturbed, dim=0)
        expanded_input = torch.cat(all_expanded_inputs, dim=0)
        partial_mask = torch.cat(all_partial_masks, dim=0)
        weights = torch.tensor(all_weights, device=input_ids.device, dtype=torch.float32)
        repeated_pocket_embeddings = torch.cat(all_pocket_embeddings, dim=0)
        repeated_pocket_mask = torch.cat(all_pocket_masks, dim=0)

        logits = score_fn(perturbed_seq, repeated_pocket_embeddings, repeated_pocket_mask)
        completion_logits = logits[:, -logits_to_keep:, :]
        completion_targets = expanded_input[:, -logits_to_keep:]
        completion_loss_mask = partial_mask[:, -logits_to_keep:]
        selective_outputs = selective_log_softmax(
            logits=completion_logits,
            index=completion_targets,
            weights=weights,
            mask=completion_loss_mask,
            return_normalized_entropy=return_normalized_entropy,
        )
        if return_normalized_entropy:
            per_token_logps, normalized_entropy = selective_outputs
            normalized_entropy = normalized_entropy.view(
                num_iterations,
                batch_size,
                logits_to_keep,
            ).permute(1, 0, 2)
        else:
            per_token_logps = selective_outputs
            normalized_entropy = None
        per_token_logps = per_token_logps.view(
            num_iterations,
            batch_size,
            logits_to_keep,
        ).permute(1, 0, 2)

    per_token_logps = per_token_logps.to(torch.float32)
    if not return_normalized_entropy:
        return per_token_logps
    return per_token_logps, normalized_entropy.to(torch.float32)


class PocketPrefixCpGRPOPolicy:
    def __init__(self, checkpoint_path, device, bf16=True, trainable=True):
        checkpoint = load_checkpoint_payload(checkpoint_path)
        require_multimodal_checkpoint(checkpoint, checkpoint_path)

        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.bf16 = bf16
        self.model = PocketPrefixGenMol.load_from_checkpoint(checkpoint_path, map_location='cpu')
        self.model.to(self.device)
        self.model.pocket_encoder.to(self.device)

        if self.model.ema:
            self.model.ema.move_shadow_params_to_device(self.device)
            self.model.ema.copy_to(self.model._ema_parameters())

        self.mask_index = self.model.mask_index
        self.bos_index = self.model.bos_index
        self.eos_index = self.model.eos_index
        self.pad_index = self.model.tokenizer.pad_token_id
        self.use_bracket_safe = bool(self.model.config.training.get('use_bracket_safe'))

        if not trainable:
            self.freeze()

    def _root_model(self):
        return _unwrap_module(self.model)

    def freeze(self):
        model = self._root_model()
        model.eval()
        for parameter in itertools.chain(model.backbone.parameters(), model.projector.parameters()):
            parameter.requires_grad = False

    def train(self):
        self._root_model().train()

    @property
    def autocast_context(self):
        if self.device.type != 'cuda':
            return nullcontext()
        if not self.bf16:
            return nullcontext()
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)

    def _unwrap_backbone(self):
        return _unwrap_module(self._root_model().backbone)

    def _unwrap_projector(self):
        return _unwrap_module(self._root_model().projector)

    def trainable_parameters(self):
        return itertools.chain(self._unwrap_backbone().parameters(), self._unwrap_projector().parameters())

    def update_ema(self):
        model = self._root_model()
        if model.ema:
            model.ema.update(self.trainable_parameters())

    def enable_gradient_checkpointing(self, gradient_checkpointing_kwargs=None):
        kwargs = gradient_checkpointing_kwargs or {}
        if hasattr(self._unwrap_backbone(), 'gradient_checkpointing_enable'):
            self._unwrap_backbone().gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)

    def get_pocket_raw_embeddings(self, pocket_coords):
        encoded = self._root_model().encode_pocket_batch(pocket_coords)
        return pad_prefix_embeddings(encoded, device=self.device, dtype=torch.float32)

    def sync_from(self, other_policy, alpha):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'alpha must be in [0, 1], got {alpha}')

        source_state = other_policy.get_trainable_state_dict()
        target_state = self.get_trainable_state_dict()
        mixed_state = {}
        for key, target_value in target_state.items():
            source_value = source_state[key].detach().to(device=target_value.device, dtype=target_value.dtype)
            target_value = target_value.detach()
            if torch.is_floating_point(target_value):
                mixed_state[key] = target_value.mul(1.0 - alpha).add(source_value, alpha=alpha)
            else:
                mixed_state[key] = source_value
        self.load_trainable_state_dict(mixed_state)

    @contextmanager
    def eval_mode(self):
        model = self._root_model()
        backbone = model.backbone
        projector = model.projector
        was_backbone_training = backbone.training
        was_projector_training = projector.training
        backbone.eval()
        projector.eval()
        try:
            yield
        finally:
            backbone.train(was_backbone_training)
            projector.train(was_projector_training)

    def forward_logits(self, input_ids, pocket_raw_embeddings, pocket_mask):
        input_ids = input_ids.clone()
        attention_mask = input_ids != self.pad_index
        with self.autocast_context:
            logits = self._root_model().forward_conditioned_logits_from_padded_raw_pocket(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pocket_raw_embeddings=pocket_raw_embeddings,
                pocket_mask=pocket_mask,
            )
        return logits.float()

    def per_token_logps(
        self,
        input_ids,
        logits_to_keep,
        completion_mask,
        mask_seeds,
        gradient_accumulation_steps,
        requires_grad,
        pocket_raw_embeddings,
        pocket_mask,
        return_normalized_entropy=False,
    ):
        def score_fn(batch, repeated_pocket_embeddings, repeated_pocket_mask):
            return self.forward_logits(batch, repeated_pocket_embeddings, repeated_pocket_mask)

        return get_per_token_logps_with_pocket(
            score_fn=score_fn,
            input_ids=input_ids,
            logits_to_keep=logits_to_keep,
            completion_mask=completion_mask,
            mask_token_id=self.mask_index,
            mask_seeds=mask_seeds,
            gradient_accumulation_steps=gradient_accumulation_steps,
            requires_grad=requires_grad,
            pocket_raw_embeddings=pocket_raw_embeddings,
            pocket_mask=pocket_mask,
            return_normalized_entropy=return_normalized_entropy,
        )

    def _decode_safe_strings(self, token_ids):
        return self._root_model().tokenizer.batch_decode(token_ids, skip_special_tokens=True)

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

    def rollout_specs(self, specs, pocket_raw_embeddings, pocket_mask, generation_batch_size, seed):
        if not specs:
            raise ValueError('specs must be non-empty')
        if generation_batch_size <= 0:
            raise ValueError('generation_batch_size must be positive')
        if pocket_raw_embeddings.size(0) != len(specs):
            raise ValueError(
                f'pocket_raw_embeddings batch size must match specs length: {pocket_raw_embeddings.size(0)} vs {len(specs)}'
            )
        if pocket_mask.size(0) != len(specs):
            raise ValueError(f'pocket_mask batch size must match specs length: {pocket_mask.size(0)} vs {len(specs)}')

        torch.manual_seed(seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)

        global_max_add_len = max(spec.add_seq_len for spec in specs)
        chunk_outputs = []
        chunk_masks = []

        with self.eval_mode():
            with torch.no_grad():
                mdlm = self._root_model().mdlm
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
                    chunk_pocket_embeddings = pocket_raw_embeddings[start:start + chunk_size]
                    chunk_pocket_mask = pocket_mask[start:start + chunk_size]
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
                    num_steps = max(mdlm.get_num_steps_confidence(x), 2)
                    for step_idx in range(num_steps):
                        logits = self.forward_logits(x, chunk_pocket_embeddings, chunk_pocket_mask)
                        x = mdlm.step_confidence(
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
        return PocketPrefixRolloutBatch(
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            full_token_ids=token_ids,
            specs=list(specs),
            safe_strings=safe_strings,
            smiles=smiles,
        )

    def load_ema_state(self, ema_state):
        model = self._root_model()
        if model.ema and ema_state is not None:
            model.ema.load_state_dict(ema_state)
            model.ema.move_shadow_params_to_device(self.device)

    def load_trainable_state_dict(self, state_dict):
        backbone_state = {}
        projector_state = {}
        for key, value in state_dict.items():
            if key.startswith('backbone.'):
                backbone_state[key[len('backbone.'):]] = value
            elif key.startswith('projector.'):
                projector_state[key[len('projector.'):]] = value
            else:
                raise ValueError(f'Unexpected trainable state key: {key}')
        self._unwrap_backbone().load_state_dict(backbone_state, strict=True)
        self._unwrap_projector().load_state_dict(projector_state, strict=True)

    def get_trainable_state_dict(self):
        state = {}
        state.update({f'backbone.{key}': value for key, value in _move_to_cpu(self._unwrap_backbone().state_dict()).items()})
        state.update({f'projector.{key}': value for key, value in _move_to_cpu(self._unwrap_projector().state_dict()).items()})
        return state

    def save_checkpoint(self, path, step, accelerator=None):
        checkpoint = load_checkpoint_payload(self.checkpoint_path)
        require_multimodal_checkpoint(checkpoint, self.checkpoint_path)

        if accelerator is None:
            model_state = _move_to_cpu(self._root_model().state_dict())
        else:
            model_state = _move_to_cpu(accelerator.get_state_dict(self.model))

        checkpoint['state_dict'] = model_state
        checkpoint['global_step'] = int(step)
        checkpoint['epoch'] = 0
        checkpoint['optimizer_states'] = []
        checkpoint['lr_schedulers'] = []

        model = self._root_model()
        if model.ema:
            checkpoint['ema'] = _move_to_cpu(model.ema.state_dict())

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
