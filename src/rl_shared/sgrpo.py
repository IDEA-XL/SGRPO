import math

import torch

VALID_SGRPO_HIERARCHIES = {'advantage_sum', 'reward_sum', 'hierarchical_sum'}
VALID_GROUP_REWRAD_CREDITS = {'broadcast', 'loo'}


def normalize_reward_thresholds(thresholds):
    if thresholds is None:
        return {}
    if not isinstance(thresholds, dict):
        raise ValueError(
            'individual_reward_thresholds must be a dict keyed by reward name or null'
        )

    normalized = {}
    for reward_name, value in thresholds.items():
        if not isinstance(reward_name, str) or not reward_name:
            raise ValueError(
                f'individual_reward_thresholds keys must be non-empty strings, got {reward_name!r}'
            )
        if value is None:
            normalized[reward_name] = None
            continue
        try:
            normalized[reward_name] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f'individual_reward_thresholds[{reward_name!r}] must be a float or null, got {value!r}'
            ) from exc
    return normalized


def validate_reward_threshold_names(thresholds, available_reward_names):
    normalized = normalize_reward_thresholds(thresholds)
    unknown = sorted(set(normalized) - set(available_reward_names))
    if unknown:
        raise ValueError(
            'individual_reward_thresholds contains unsupported reward names: '
            f'{unknown}; expected subset of {sorted(available_reward_names)}'
        )
    return normalized


def _compute_group_reward_indicator(group_mean_individual_rewards, group_reward_thresholds, num_groups, device):
    indicator = torch.ones(num_groups, device=device, dtype=torch.float32)
    active_threshold_count = 0
    for reward_name, threshold in group_reward_thresholds.items():
        if threshold is None:
            continue
        if reward_name not in group_mean_individual_rewards:
            raise ValueError(
                f'missing group mean individual reward tensor for thresholded reward {reward_name!r}'
            )
        reward_means = group_mean_individual_rewards[reward_name]
        if reward_means.dim() != 1:
            raise ValueError(f'group mean individual rewards for {reward_name!r} must be 1D')
        if reward_means.numel() != num_groups:
            raise ValueError(
                'group mean individual reward tensor length must match number of groups: '
                f'{reward_name!r} has {reward_means.numel()} vs {num_groups}'
            )
        indicator = indicator * (reward_means > float(threshold)).to(dtype=torch.float32)
        active_threshold_count += 1
    return indicator, active_threshold_count


def compute_grouped_advantages(
    rewards,
    num_generations,
    scale_rewards=False,
    sample_mask=None,
):
    if rewards.dim() != 1:
        raise ValueError('rewards must be a 1D tensor')
    if num_generations <= 1:
        raise ValueError('num_generations must be greater than 1')
    if rewards.numel() % num_generations != 0:
        raise ValueError('rewards length must be divisible by num_generations')

    rewards_grouped = rewards.view(-1, num_generations)
    if sample_mask is None:
        sum_group = rewards_grouped.sum(dim=1, keepdim=True)
        baseline = (sum_group - rewards_grouped) / (num_generations - 1)
        advantages = (rewards_grouped - baseline).view(-1)
        std_grouped = rewards_grouped.std(dim=1, keepdim=True)
        repeated_std = std_grouped.repeat_interleave(num_generations, dim=1).view(-1)
        if scale_rewards:
            advantages = advantages / (repeated_std + 1e-4)
        zero_std_ratio = (repeated_std < 1e-6).to(torch.float32).mean().item()
        return advantages, repeated_std, zero_std_ratio

    if not torch.is_tensor(sample_mask):
        raise ValueError('sample_mask must be a tensor when provided')
    if sample_mask.dim() != 1 or sample_mask.shape != rewards.shape:
        raise ValueError(
            'sample_mask must be a 1D tensor matching rewards, '
            f'got {tuple(sample_mask.shape)} vs {tuple(rewards.shape)}'
        )
    mask_grouped = sample_mask.to(device=rewards.device, dtype=torch.bool).view(
        -1,
        num_generations,
    )
    active_counts = mask_grouped.sum(dim=1, keepdim=True)

    masked_rewards = torch.where(
        mask_grouped,
        rewards_grouped,
        torch.zeros_like(rewards_grouped),
    )
    sum_group = masked_rewards.sum(dim=1, keepdim=True)
    loo_denominator = (active_counts - 1).clamp(min=1).to(dtype=rewards.dtype)
    baseline = (sum_group - rewards_grouped) / loo_denominator
    grouped_advantages = torch.where(
        mask_grouped & (active_counts > 1),
        rewards_grouped - baseline,
        torch.zeros_like(rewards_grouped),
    )

    active_counts_float = active_counts.to(dtype=rewards.dtype)
    means = sum_group / active_counts_float.clamp(min=1.0)
    squared_deviations = torch.where(
        mask_grouped,
        (rewards_grouped - means).square(),
        torch.zeros_like(rewards_grouped),
    ).sum(dim=1, keepdim=True)
    variances = torch.where(
        active_counts > 1,
        squared_deviations / (active_counts_float - 1).clamp(min=1.0),
        torch.zeros_like(squared_deviations),
    )
    std_grouped = variances.sqrt()
    repeated_std_grouped = torch.where(
        mask_grouped,
        std_grouped.expand_as(rewards_grouped),
        torch.zeros_like(rewards_grouped),
    )
    advantages = grouped_advantages.view(-1)
    repeated_std = repeated_std_grouped.view(-1)
    if scale_rewards:
        advantages = torch.where(
            sample_mask.to(device=rewards.device, dtype=torch.bool),
            advantages / (repeated_std + 1e-4),
            torch.zeros_like(advantages),
        )
    active_std = repeated_std[sample_mask.to(device=rewards.device, dtype=torch.bool)]
    zero_std_ratio = (
        1.0
        if active_std.numel() == 0
        else (active_std < 1e-6).to(torch.float32).mean().item()
    )
    return advantages, repeated_std, zero_std_ratio


def compute_hierarchical_sum_advantages(
    rollout_rewards,
    group_rewards,
    num_generations,
    supergroup_num_groups,
    group_advantage_weight,
    scale_rewards=False,
):
    if rollout_rewards.dim() != 1:
        raise ValueError('rollout_rewards must be a 1D tensor')
    if group_rewards.dim() != 1:
        raise ValueError('group_rewards must be a 1D tensor')
    if num_generations <= 1:
        raise ValueError('num_generations must be greater than 1')
    if supergroup_num_groups <= 1:
        raise ValueError('supergroup_num_groups must be greater than 1')
    if not 0.0 <= group_advantage_weight <= 1.0:
        raise ValueError('group_advantage_weight must be in [0, 1]')
    if rollout_rewards.numel() % num_generations != 0:
        raise ValueError('rollout_rewards length must be divisible by num_generations')

    num_groups = rollout_rewards.numel() // num_generations
    if group_rewards.numel() != num_groups:
        raise ValueError(
            f'group_rewards length must equal number of groups: {group_rewards.numel()} vs {num_groups}'
        )
    if num_groups % supergroup_num_groups != 0:
        raise ValueError(
            'number of groups must be divisible by supergroup_num_groups: '
            f'{num_groups} vs {supergroup_num_groups}'
        )

    group_advantage_weight = float(group_advantage_weight)
    rollout_advantage_weight = 1.0 - group_advantage_weight

    expanded_group_rewards = group_rewards.repeat_interleave(num_generations)
    combined_rewards = (
        rollout_rewards * rollout_advantage_weight
        + expanded_group_rewards * group_advantage_weight
    )

    rollout_rewards_grouped = rollout_rewards.view(num_groups, num_generations)
    rollout_baseline = (
        (rollout_rewards_grouped.sum(dim=1, keepdim=True) - rollout_rewards_grouped)
        / (num_generations - 1)
    ).view(-1)
    group_rewards_grouped = group_rewards.view(-1, supergroup_num_groups)
    group_baseline = (
        (group_rewards_grouped.sum(dim=1, keepdim=True) - group_rewards_grouped)
        / (supergroup_num_groups - 1)
    ).view(-1).repeat_interleave(num_generations)
    baseline = rollout_baseline * rollout_advantage_weight + group_baseline * group_advantage_weight
    advantages = combined_rewards - baseline

    rollout_supergroup_size = num_generations * supergroup_num_groups
    final_reward_std = (
        combined_rewards.view(-1, rollout_supergroup_size)
        .std(dim=1, keepdim=True)
        .repeat_interleave(rollout_supergroup_size, dim=1)
        .view(-1)
    )
    if scale_rewards:
        advantages = advantages / (final_reward_std + 1e-4)
    zero_std_ratio = (final_reward_std < 1e-6).to(torch.float32).mean().item()
    return advantages, final_reward_std, zero_std_ratio, combined_rewards.mean().item()


def _normalize_group_reward_credits(group_reward_credits, num_groups, num_generations, device):
    if group_reward_credits is None:
        raise ValueError('group_reward_credits is required when group_rewrad_credit=loo')
    if not torch.is_tensor(group_reward_credits):
        raise ValueError('group_reward_credits must be a tensor')
    credits = group_reward_credits.to(device=device, dtype=torch.float32)
    if credits.dim() == 1:
        if credits.numel() != num_groups * num_generations:
            raise ValueError(
                'flat group_reward_credits length must equal num_groups * num_generations: '
                f'{credits.numel()} vs {num_groups * num_generations}'
            )
        credits = credits.view(num_groups, num_generations)
    elif credits.dim() == 2:
        if credits.shape != (num_groups, num_generations):
            raise ValueError(
                'group_reward_credits must have shape '
                f'[{num_groups}, {num_generations}], got {list(credits.shape)}'
            )
    else:
        raise ValueError('group_reward_credits must be 1D or 2D')
    return credits


def _credit_weights(group_reward_credits, temperature):
    if temperature <= 0.0:
        raise ValueError('group_rewrad_credit_temperature must be positive')
    centered = group_reward_credits - group_reward_credits.mean(dim=1, keepdim=True)
    std = group_reward_credits.std(dim=1, keepdim=True)
    z_scores = centered / (std + 1e-4)
    num_generations = group_reward_credits.size(1)
    positive_weights = num_generations * torch.softmax(z_scores / float(temperature), dim=1)
    negative_weights = num_generations * torch.softmax(-z_scores / float(temperature), dim=1)
    return positive_weights, negative_weights


def _apply_loo_credit_to_group_advantages(group_advantages, group_reward_credits, temperature):
    positive_weights, negative_weights = _credit_weights(group_reward_credits, temperature)
    group_advantages = group_advantages.view(-1, 1)
    credited = (
        torch.clamp(group_advantages, min=0.0) * positive_weights
        - torch.clamp(-group_advantages, min=0.0) * negative_weights
    )
    return credited.view(-1)


def _allocate_group_rewards_with_loo_credit(
    group_rewards,
    group_advantages,
    group_reward_credits,
    temperature,
):
    # Preserve each group's mean reward while using the signed supergroup
    # advantage to decide which LOO-credit side receives the reward residual.
    positive_weights, negative_weights = _credit_weights(group_reward_credits, temperature)
    group_rewards = group_rewards.view(-1, 1)
    group_advantages = group_advantages.view(-1, 1)
    allocation_delta = (
        torch.clamp(group_advantages, min=0.0) * (positive_weights - 1.0)
        - torch.clamp(-group_advantages, min=0.0) * (negative_weights - 1.0)
    )
    allocated = group_rewards + allocation_delta
    return allocated.view(-1)


def compute_sgrpo_advantages(
    rollout_rewards,
    group_rewards,
    num_generations,
    supergroup_num_groups,
    group_advantage_weight,
    scale_rewards=False,
    hierarchy='advantage_sum',
    group_mean_individual_rewards=None,
    individual_reward_thresholds=None,
    group_rewrad_credit='broadcast',
    group_rewrad_credit_temperature=1.0,
    group_reward_credits=None,
):
    if rollout_rewards.dim() != 1:
        raise ValueError('rollout_rewards must be a 1D tensor')
    if group_rewards.dim() != 1:
        raise ValueError('group_rewards must be a 1D tensor')
    if num_generations <= 1:
        raise ValueError('num_generations must be greater than 1')
    if supergroup_num_groups <= 1:
        raise ValueError('supergroup_num_groups must be greater than 1')
    if not 0.0 <= group_advantage_weight <= 1.0:
        raise ValueError('group_advantage_weight must be in [0, 1]')
    if hierarchy not in VALID_SGRPO_HIERARCHIES:
        raise ValueError(
            f"hierarchy must be one of {sorted(VALID_SGRPO_HIERARCHIES)}, got {hierarchy!r}"
        )
    if group_rewrad_credit not in VALID_GROUP_REWRAD_CREDITS:
        raise ValueError(
            f"group_rewrad_credit must be one of {sorted(VALID_GROUP_REWRAD_CREDITS)}, "
            f'got {group_rewrad_credit!r}'
        )
    group_rewrad_credit_temperature = float(group_rewrad_credit_temperature)
    if group_rewrad_credit_temperature <= 0.0:
        raise ValueError('group_rewrad_credit_temperature must be positive')
    if rollout_rewards.numel() % num_generations != 0:
        raise ValueError('rollout_rewards length must be divisible by num_generations')

    num_groups = rollout_rewards.numel() // num_generations
    if group_rewards.numel() != num_groups:
        raise ValueError(
            f'group_rewards length must equal number of groups: {group_rewards.numel()} vs {num_groups}'
        )
    if num_groups % supergroup_num_groups != 0:
        raise ValueError(
            'number of groups must be divisible by supergroup_num_groups: '
            f'{num_groups} vs {supergroup_num_groups}'
        )

    group_mean_individual_rewards = (
        {} if group_mean_individual_rewards is None else dict(group_mean_individual_rewards)
    )
    group_reward_thresholds = normalize_reward_thresholds(individual_reward_thresholds)
    group_reward_indicator, active_threshold_count = _compute_group_reward_indicator(
        group_mean_individual_rewards=group_mean_individual_rewards,
        group_reward_thresholds=group_reward_thresholds,
        num_groups=num_groups,
        device=group_rewards.device,
    )
    gated_group_rewards = group_rewards * group_reward_indicator

    rollout_supergroup_size = num_generations * supergroup_num_groups
    rollout_advantages, rollout_reward_std, rollout_zero_std_ratio = compute_grouped_advantages(
        rewards=rollout_rewards,
        num_generations=rollout_supergroup_size,
        scale_rewards=scale_rewards,
    )
    group_advantages, group_reward_std, group_zero_std_ratio = compute_grouped_advantages(
        rewards=gated_group_rewards,
        num_generations=supergroup_num_groups,
        scale_rewards=scale_rewards,
    )
    if group_rewrad_credit == 'loo':
        normalized_group_reward_credits = _normalize_group_reward_credits(
            group_reward_credits=group_reward_credits,
            num_groups=num_groups,
            num_generations=num_generations,
            device=group_rewards.device,
        )
        expanded_group_advantages = _apply_loo_credit_to_group_advantages(
            group_advantages=group_advantages,
            group_reward_credits=normalized_group_reward_credits,
            temperature=group_rewrad_credit_temperature,
        )
        expanded_group_rewards = _allocate_group_rewards_with_loo_credit(
            group_rewards=gated_group_rewards,
            group_advantages=group_advantages,
            group_reward_credits=normalized_group_reward_credits,
            temperature=group_rewrad_credit_temperature,
        )
    else:
        normalized_group_reward_credits = None
        expanded_group_advantages = group_advantages.repeat_interleave(num_generations)
        expanded_group_rewards = gated_group_rewards.repeat_interleave(num_generations)
    rollout_advantage_weight = 1.0 - group_advantage_weight
    if hierarchy == 'advantage_sum':
        final_advantages = (
            rollout_advantages * rollout_advantage_weight
            + expanded_group_advantages * group_advantage_weight
        )
        final_reward_std = rollout_reward_std
        final_zero_std_ratio = rollout_zero_std_ratio
        combined_reward_mean = float('nan')
    elif hierarchy == 'reward_sum':
        combined_rewards = (
            rollout_rewards * rollout_advantage_weight
            + expanded_group_rewards * group_advantage_weight
        )
        final_advantages, final_reward_std, final_zero_std_ratio = compute_grouped_advantages(
            rewards=combined_rewards,
            num_generations=rollout_supergroup_size,
            scale_rewards=scale_rewards,
        )
        combined_reward_mean = combined_rewards.mean().item()
    else:
        if group_rewrad_credit == 'loo':
            combined_rewards = (
                rollout_rewards * rollout_advantage_weight
                + expanded_group_rewards * group_advantage_weight
            )
            rollout_rewards_grouped = rollout_rewards.view(num_groups, num_generations)
            rollout_baseline = (
                (rollout_rewards_grouped.sum(dim=1, keepdim=True) - rollout_rewards_grouped)
                / (num_generations - 1)
            ).view(-1)
            group_rewards_grouped = gated_group_rewards.view(-1, supergroup_num_groups)
            group_baseline = (
                (group_rewards_grouped.sum(dim=1, keepdim=True) - group_rewards_grouped)
                / (supergroup_num_groups - 1)
            ).view(-1).repeat_interleave(num_generations)
            baseline = rollout_baseline * rollout_advantage_weight + group_baseline * group_advantage_weight
            final_advantages = combined_rewards - baseline
            final_reward_std = (
                combined_rewards.view(-1, rollout_supergroup_size)
                .std(dim=1, keepdim=True)
                .repeat_interleave(rollout_supergroup_size, dim=1)
                .view(-1)
            )
            if scale_rewards:
                final_advantages = final_advantages / (final_reward_std + 1e-4)
            final_zero_std_ratio = (final_reward_std < 1e-6).to(torch.float32).mean().item()
            combined_reward_mean = combined_rewards.mean().item()
        else:
            final_advantages, final_reward_std, final_zero_std_ratio, combined_reward_mean = (
                compute_hierarchical_sum_advantages(
                    rollout_rewards=rollout_rewards,
                    group_rewards=gated_group_rewards,
                    num_generations=num_generations,
                    supergroup_num_groups=supergroup_num_groups,
                    group_advantage_weight=group_advantage_weight,
                    scale_rewards=scale_rewards,
                )
            )

    metrics = {
        'rollout_advantage_mean': rollout_advantages.mean().item(),
        'group_advantage_mean': expanded_group_advantages.mean().item(),
        'group_reward_mean': gated_group_rewards.mean().item(),
        'group_reward_raw_mean': group_rewards.mean().item(),
        'group_advantage_weight': float(group_advantage_weight),
        'rollout_zero_std_ratio': final_zero_std_ratio,
        'group_zero_std_ratio': group_zero_std_ratio,
        'rollout_reward_std_mean': final_reward_std.mean().item(),
        'group_reward_std_mean': group_reward_std.mean().item(),
        'group_reward_indicator_mean': group_reward_indicator.mean().item(),
        'active_threshold_count': float(active_threshold_count),
        'hierarchy_reward_sum_enabled': 1.0 if hierarchy == 'reward_sum' else 0.0,
        'hierarchy_hierarchical_sum_enabled': 1.0 if hierarchy == 'hierarchical_sum' else 0.0,
        'combined_reward_mean': combined_reward_mean,
        'group_rewrad_credit_loo_enabled': 1.0 if group_rewrad_credit == 'loo' else 0.0,
        'group_rewrad_credit_temperature': float(group_rewrad_credit_temperature),
    }
    if normalized_group_reward_credits is not None:
        metrics['group_rewrad_credit_mean'] = normalized_group_reward_credits.mean().item()
        metrics['group_rewrad_credit_std'] = normalized_group_reward_credits.std().item()
    return final_advantages, expanded_group_advantages, rollout_advantages, metrics


def compute_group_reward_regularizer_advantages(group_rewards, num_generations, scale_rewards=False):
    if group_rewards.dim() != 1:
        raise ValueError('group_rewards must be a 1D tensor')
    if num_generations <= 0:
        raise ValueError('num_generations must be positive')

    num_groups = int(group_rewards.numel())
    if num_groups <= 1:
        raise ValueError('group_rewards must contain at least two groups for a centered regularizer signal')

    group_advantages, group_reward_std, group_zero_std_ratio = compute_grouped_advantages(
        rewards=group_rewards,
        num_generations=num_groups,
        scale_rewards=scale_rewards,
    )
    expanded_group_advantages = group_advantages.repeat_interleave(num_generations)
    metrics = {
        'group_advantage_mean': expanded_group_advantages.mean().item(),
        'group_reward_mean': group_rewards.mean().item(),
        'group_zero_std_ratio': group_zero_std_ratio,
        'group_reward_std_mean': group_reward_std.mean().item(),
    }
    return expanded_group_advantages, metrics


def compute_clipped_grpo_loss(
    new_log_probs,
    old_log_probs,
    advantages,
    completion_mask,
    epsilon,
    ref_log_probs=None,
    beta=0.0,
):
    if new_log_probs.shape != old_log_probs.shape:
        raise ValueError('new_log_probs and old_log_probs must have the same shape')
    if new_log_probs.dim() != 3 or new_log_probs.shape[1] != 1:
        raise ValueError('expected log prob tensors to have shape [batch, 1, seq_len]')
    if completion_mask.dim() != 2:
        raise ValueError('completion_mask must have shape [batch, seq_len]')

    completion_mask = completion_mask.to(dtype=torch.float32)
    advantages = advantages.view(-1, 1, 1)
    coef_1 = torch.exp(new_log_probs - old_log_probs)
    coef_2 = torch.clamp(coef_1, 1 - epsilon, 1 + epsilon)
    loss_1 = coef_1 * advantages
    loss_2 = coef_2 * advantages
    per_token_loss = -torch.min(loss_1, loss_2)

    kl_value = None
    if ref_log_probs is not None and beta != 0.0:
        per_token_kl = torch.exp(ref_log_probs - new_log_probs) - (ref_log_probs - new_log_probs) - 1.0
        per_token_loss = per_token_loss + beta * per_token_kl
        kl_value = (per_token_kl[:, 0, :] * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)

    loss = (per_token_loss[:, 0, :] * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)

    is_low_clipped = (coef_1 < 1 - epsilon) & (advantages < 0)
    is_high_clipped = (coef_1 > 1 + epsilon) & (advantages > 0)
    is_region_clipped = is_low_clipped | is_high_clipped

    low_clip = (is_low_clipped[:, 0, :].to(torch.float32) * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
    high_clip = (is_high_clipped[:, 0, :].to(torch.float32) * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
    region_clip = (is_region_clipped[:, 0, :].to(torch.float32) * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
    ratio_mean = (coef_1[:, 0, :] * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)

    metrics = {
        'ratio_mean': ratio_mean.detach(),
        'clip_ratio_low_mean': low_clip.detach(),
        'clip_ratio_high_mean': high_clip.detach(),
        'clip_ratio_region_mean': region_clip.detach(),
    }
    if kl_value is not None:
        metrics['kl_mean'] = kl_value.detach()

    return loss, metrics


def compute_warmup_steps(max_steps, warmup_ratio):
    return max(1, int(math.ceil(max_steps * warmup_ratio)))
