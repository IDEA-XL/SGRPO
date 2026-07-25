import math
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from rl_shared.sgrpo import (
    VALID_GROUP_REWRAD_CREDITS,
    VALID_SGRPO_HIERARCHIES,
    compute_clipped_grpo_loss,
    compute_grouped_advantages,
    compute_group_reward_regularizer_advantages,
    compute_sgrpo_advantages,
    compute_warmup_steps,
    normalize_reward_thresholds,
    validate_reward_threshold_names,
)


def _cpu_generator(seed):
    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(seed))
    return generator


def split_tensor_dict(tensor_dict, num_chunks):
    first_tensor = next(value for value in tensor_dict.values() if isinstance(value, torch.Tensor))
    if first_tensor.shape[0] % num_chunks != 0:
        raise ValueError('tensor batch dimension must be divisible by num_chunks')

    chunk_size = first_tensor.shape[0] // num_chunks
    chunks = []
    for chunk_idx in range(num_chunks):
        chunk = {}
        start = chunk_idx * chunk_size
        end = (chunk_idx + 1) * chunk_size
        for key, value in tensor_dict.items():
            if isinstance(value, torch.Tensor) and value.shape[:1] == first_tensor.shape[:1]:
                chunk[key] = value[start:end]
            else:
                chunk[key] = value
        chunks.append(chunk)
    return chunks


def selective_log_softmax(
    logits,
    index,
    weights=None,
    mask=None,
    return_normalized_entropy=False,
):
    full_batch_size = logits.size(0) // 3
    if full_batch_size == 0:
        raise ValueError('logits batch size must be at least 3')

    if weights is None or mask is None:
        raise ValueError('weights and mask are required for coupled selective_log_softmax')

    num_iterations = weights.size(0) // 3
    batch_size = full_batch_size // num_iterations
    per_token_logps = []
    per_token_entropies = []
    entropy_normalizer = math.log(logits.size(-1))
    if return_normalized_entropy and entropy_normalizer <= 0.0:
        raise ValueError('entropy requires a vocabulary with at least two tokens')

    for sample_idx in range(full_batch_size):
        labels = index[sample_idx].clone()
        chunk_idx, offset = divmod(sample_idx, batch_size)
        base = chunk_idx * 3 * batch_size
        logits_index = torch.tensor(
            [base + offset, base + batch_size + offset, base + 2 * batch_size + offset],
            device=logits.device,
        )
        seq_logits = logits[logits_index]
        seq_logps = F.log_softmax(seq_logits, dim=-1)
        gathered = seq_logps.gather(
            dim=-1,
            index=labels.unsqueeze(0).unsqueeze(-1).expand(3, -1, 1),
        ).squeeze(-1)

        seq_weights = weights[chunk_idx * 3:(chunk_idx + 1) * 3]
        seq_mask = mask[sample_idx].clone()
        weighted = torch.where(seq_mask, gathered[1] * seq_weights[1], gathered[2] * seq_weights[2])
        per_token_logps.append((gathered[0] + weighted) / 2)

        if return_normalized_entropy:
            seq_entropies = -(seq_logps.exp() * seq_logps).sum(dim=-1)
            weighted_entropy = torch.where(
                seq_mask,
                seq_entropies[1] * seq_weights[1],
                seq_entropies[2] * seq_weights[2],
            )
            per_token_entropies.append(
                ((seq_entropies[0] + weighted_entropy) / 2) / entropy_normalizer
            )

    stacked_logps = torch.stack(per_token_logps, dim=0)
    if not return_normalized_entropy:
        return stacked_logps
    return stacked_logps, torch.stack(per_token_entropies, dim=0)


def forward_process(
    batch,
    completion_mask,
    mask_id,
    seed,
    gradient_accumulation_steps=1,
    accumulate=False,
):
    generator = _cpu_generator(seed)
    batch_size, seq_len = batch.shape
    mask_ratio = 0.2 + 0.6 * torch.rand((), generator=generator).item()

    if accumulate:
        if batch_size % gradient_accumulation_steps != 0:
            raise ValueError('batch size must be divisible by gradient_accumulation_steps when accumulate=True')
        random_matrix = torch.rand((batch_size // gradient_accumulation_steps, seq_len), generator=generator)
        random_matrix = torch.cat([random_matrix] * gradient_accumulation_steps, dim=0)
    else:
        random_matrix = torch.rand((batch_size, seq_len), generator=generator)
    random_matrix = random_matrix.to(device=batch.device, dtype=torch.float32)

    full_mask = completion_mask
    mask_a = completion_mask & (random_matrix < mask_ratio)
    mask_b = completion_mask & (random_matrix > mask_ratio)
    noisy_batch = [
        torch.where(full_mask, mask_id, batch),
        torch.where(mask_a, mask_id, batch),
        torch.where(mask_b, mask_id, batch),
    ]
    return noisy_batch, [1.0, 1.0 / mask_ratio, 1.0 / (1.0 - mask_ratio)], mask_a


def get_per_token_logps(
    score_fn,
    input_ids,
    logits_to_keep,
    completion_mask,
    mask_token_id,
    mask_seeds,
    gradient_accumulation_steps,
    requires_grad,
    return_normalized_entropy=False,
):
    if input_ids.dim() != 3:
        raise ValueError(f'Expected input_ids to have 3 dimensions, got {input_ids.dim()}')
    if completion_mask.dim() != 2:
        raise ValueError(f'Expected completion_mask to have 2 dimensions, got {completion_mask.dim()}')

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

    prompt_length = seq_len - logits_to_keep
    full_completion_mask = torch.zeros((batch_size, seq_len), device=input_ids.device, dtype=torch.bool)
    full_completion_mask[:, prompt_length:] = completion_mask

    grad_context = nullcontext() if requires_grad else torch.no_grad()
    with grad_context:
        all_perturbed = []
        all_weights = []
        all_expanded_inputs = []
        all_partial_masks = []

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

        perturbed_seq = torch.cat(all_perturbed, dim=0)
        expanded_input = torch.cat(all_expanded_inputs, dim=0)
        partial_mask = torch.cat(all_partial_masks, dim=0)
        weights = torch.tensor(all_weights, device=input_ids.device, dtype=torch.float32)

        logits = score_fn(perturbed_seq)
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
