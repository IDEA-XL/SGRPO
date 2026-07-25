import os
import random
import math
from dataclasses import dataclass

import torch

from progen2.modeling.wrapper import OfficialProGen2CausalLM


@dataclass
class ProGen2RolloutBatch:
    prompt_texts: list[str]
    prompt_lengths: torch.Tensor
    full_token_ids: torch.Tensor
    full_attention_mask: torch.Tensor
    generated_mask: torch.Tensor
    decoded_texts: list[str]
    protein_sequences: list[str]


def _extract_protein_sequence(decoded_text):
    decoded_text = str(decoded_text)
    start_idx = 1 if decoded_text and decoded_text[0] in {'1', '2'} else 0
    end_idx = len(decoded_text)
    for idx in range(start_idx, len(decoded_text)):
        if decoded_text[idx] in {'1', '2'}:
            end_idx = idx
            break
    return decoded_text[start_idx:end_idx].strip().upper()


class ProGen2Policy:
    def __init__(self, model: OfficialProGen2CausalLM, trainable: bool):
        self.model = model
        self.device = model.device
        if not trainable:
            self.freeze()

    def freeze(self):
        self.model.eval()
        for parameter in self.model.trainable_parameters():
            parameter.requires_grad = False

    def train(self):
        self.model.train()

    def generate_rollouts(
        self,
        prompt_texts,
        *,
        num_return_sequences,
        max_new_tokens,
        top_p,
        temperature,
        seed,
    ):
        if not prompt_texts:
            raise ValueError('prompt_texts must be non-empty')
        if num_return_sequences <= 0:
            raise ValueError('num_return_sequences must be positive')
        if max_new_tokens <= 0:
            raise ValueError('max_new_tokens must be positive')
        if not 0.0 < top_p <= 1.0:
            raise ValueError('top_p must be in (0, 1]')
        if temperature <= 0.0:
            raise ValueError('temperature must be positive')

        random.seed(seed)
        torch.manual_seed(seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)

        batch = self.model.tokenizer.batch_encode(prompt_texts, device=self.device)
        prompt_lengths = batch.attention_mask.sum(dim=1, dtype=torch.long)

        with torch.no_grad():
            generated = self.model.generate(
                batch.input_ids,
                attention_mask=batch.attention_mask,
                do_sample=True,
                top_p=float(top_p),
                temperature=float(temperature),
                max_new_tokens=int(max_new_tokens),
                num_return_sequences=int(num_return_sequences),
            )

        repeated_prompt_lengths = prompt_lengths.repeat_interleave(num_return_sequences)
        repeated_prompt_texts = []
        for prompt in prompt_texts:
            repeated_prompt_texts.extend([prompt] * num_return_sequences)

        full_attention_mask = generated != self.model.tokenizer.pad_token_id
        position_ids = torch.arange(generated.size(1), device=generated.device).view(1, -1)
        generated_mask = (position_ids >= repeated_prompt_lengths.unsqueeze(1)) & full_attention_mask
        decoded = self.model.tokenizer.decode_batch(generated.detach().cpu().tolist())
        protein_sequences = [_extract_protein_sequence(text) for text in decoded]
        return ProGen2RolloutBatch(
            prompt_texts=repeated_prompt_texts,
            prompt_lengths=repeated_prompt_lengths,
            full_token_ids=generated,
            full_attention_mask=full_attention_mask.to(dtype=torch.long),
            generated_mask=generated_mask,
            decoded_texts=decoded,
            protein_sequences=protein_sequences,
        )

    def per_token_logps(
        self,
        full_token_ids,
        full_attention_mask,
        generated_mask,
        requires_grad,
        return_normalized_entropy=False,
    ):
        if full_token_ids.dim() != 2:
            raise ValueError('full_token_ids must have shape [batch, seq_len]')
        if full_attention_mask.shape != full_token_ids.shape:
            raise ValueError('full_attention_mask must match full_token_ids shape')
        if generated_mask.shape != full_token_ids.shape:
            raise ValueError('generated_mask must match full_token_ids shape')

        context = self.model.autocast_context if requires_grad else torch.no_grad()
        with context:
            outputs = self.model.forward(
                input_ids=full_token_ids,
                attention_mask=full_attention_mask,
            )
            logits = outputs.logits
        shifted_logits = logits[:, :-1, :]
        shifted_targets = full_token_ids[:, 1:]
        shifted_mask = generated_mask[:, 1:] & (full_attention_mask[:, 1:] > 0)
        log_probs = torch.log_softmax(shifted_logits.float(), dim=-1)
        gathered = log_probs.gather(dim=-1, index=shifted_targets.unsqueeze(-1)).squeeze(-1)
        if not return_normalized_entropy:
            return gathered.unsqueeze(1), shifted_mask

        entropy_normalizer = math.log(log_probs.size(-1))
        if entropy_normalizer <= 0.0:
            raise ValueError('entropy requires a vocabulary with at least two tokens')
        normalized_entropy = (
            -(log_probs.exp() * log_probs).sum(dim=-1) / entropy_normalizer
        )
        return gathered.unsqueeze(1), shifted_mask, normalized_entropy.unsqueeze(1)
