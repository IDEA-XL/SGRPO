from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import yaml

from progen2.rl.trainer import ProGen2SGRPOTrainer, load_config


class _FakeAccelerator:
    def __init__(self, *, process_index, num_processes, gradient_accumulation_steps):
        self.process_index = process_index
        self.num_processes = num_processes
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.backward_losses = []
        self.no_sync_calls = 0

    def backward(self, loss):
        self.backward_losses.append(float(loss.detach().item()))
        (loss / self.gradient_accumulation_steps).backward()

    def no_sync(self, model):
        self.no_sync_calls += 1
        return nullcontext()

    def clip_grad_norm_(self, parameters, max_grad_norm):
        return torch.nn.utils.clip_grad_norm_(list(parameters), max_grad_norm)


class _FakePolicyModel:
    def __init__(self, parameter):
        self.parameter = parameter
        self.model = self

    def trainable_parameters(self):
        return [self.parameter]


def _make_seed_mapping_trainer(*, process_index, num_processes, accumulation_steps):
    trainer = ProGen2SGRPOTrainer.__new__(ProGen2SGRPOTrainer)
    trainer.config = SimpleNamespace(
        seed=42,
        gradient_accumulation_steps=accumulation_steps,
        per_device_prompt_batch_size=2,
        reward_calibration_prompt_batch_size=8,
    )
    trainer.accelerator = SimpleNamespace(
        process_index=process_index,
        num_processes=num_processes,
    )
    trainer.effective_world_size = num_processes * accumulation_steps
    trainer.prompts = list(range(128))
    trainer.global_step = 3
    trainer._prompt_cursor = trainer._prompt_cursor_after_steps(trainer.global_step)
    return trainer


def _base_config():
    return {
        'model_variant': 'progen2_sgrpo',
        'official_code_dir': '/tmp/official',
        'tokenizer_path': '/tmp/tokenizer.json',
        'init_checkpoint_dir': '/tmp/checkpoint',
        'prompt_path': '/tmp/prompts.txt',
        'per_device_prompt_batch_size': 2,
        'num_generations': 4,
        'supergroup_num_groups': 1,
        'rl_algorithm': 'grpo',
        'rewards': {
            'naturalness': {'model_name': 'esm2_t33_650M_UR50D'},
            'foldability': {},
            'stability': {
                'model_name_or_path': '/tmp/temberture',
                'base_model_name_or_path': '/tmp/protbert',
            },
            'developability': {'model_name_or_path': '/tmp/proteinsol'},
        },
    }


def test_four_gpu_two_micro_batches_match_eight_gpu_virtual_ranks():
    four_gpu = _make_seed_mapping_trainer(
        process_index=1,
        num_processes=4,
        accumulation_steps=2,
    )
    eight_gpu_rank_two = _make_seed_mapping_trainer(
        process_index=2,
        num_processes=8,
        accumulation_steps=1,
    )
    eight_gpu_rank_three = _make_seed_mapping_trainer(
        process_index=3,
        num_processes=8,
        accumulation_steps=1,
    )

    assert four_gpu._next_prompt_batch(0) == eight_gpu_rank_two._next_prompt_batch(0)
    assert four_gpu._next_prompt_batch(1) == eight_gpu_rank_three._next_prompt_batch(0)
    assert four_gpu._micro_batch_seed(0) == eight_gpu_rank_two._micro_batch_seed(0)
    assert four_gpu._micro_batch_seed(1) == eight_gpu_rank_three._micro_batch_seed(0)
    assert four_gpu._selection_seed(0) == eight_gpu_rank_two._selection_seed(0)
    assert four_gpu._selection_seed(1) == eight_gpu_rank_three._selection_seed(0)
    assert four_gpu._calibration_seed(3, 0) == eight_gpu_rank_two._calibration_seed(3, 0)
    assert four_gpu._calibration_seed(3, 1) == eight_gpu_rank_three._calibration_seed(3, 0)
    assert (
        four_gpu._calibration_prompt_cursor(0)
        == eight_gpu_rank_two._calibration_prompt_cursor(0)
    )
    assert (
        four_gpu._calibration_prompt_cursor(1)
        == eight_gpu_rank_three._calibration_prompt_cursor(0)
    )
    assert eight_gpu_rank_two._next_prompt_batch(0) == [10, 11]
    assert eight_gpu_rank_two._micro_batch_seed(0) == 45
    assert eight_gpu_rank_two._selection_seed(0) == 33033


def test_calibration_uses_all_virtual_rank_slots_in_order():
    trainer = ProGen2SGRPOTrainer.__new__(ProGen2SGRPOTrainer)
    trainer.config = SimpleNamespace(
        seed=42,
        gradient_accumulation_steps=2,
        reward_calibration_size=4,
        reward_calibration_prompt_batch_size=2,
    )
    trainer.accelerator = SimpleNamespace(
        process_index=0,
        num_processes=1,
        is_main_process=True,
    )
    trainer.effective_world_size = 2
    trainer.prompts = list(range(16))
    trainer._broadcast_object = lambda payload: payload
    trainer._all_gather_object = lambda payload: [payload]
    trainer._generate_rollouts = lambda prompts, num_return_sequences, seed: (
        SimpleNamespace(
            protein_sequences=[
                f'prompt={prompt},seed={seed}' for prompt in prompts
            ]
        )
    )

    sequences = trainer._calibration_sequences()

    assert sequences == [
        'prompt=0,seed=42',
        'prompt=1,seed=42',
        'prompt=2,seed=43',
        'prompt=3,seed=43',
    ]


def test_train_accumulates_two_micro_batches_before_one_optimizer_step(tmp_path):
    trainer = ProGen2SGRPOTrainer.__new__(ProGen2SGRPOTrainer)
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    accelerator = _FakeAccelerator(
        process_index=0,
        num_processes=4,
        gradient_accumulation_steps=2,
    )
    trainer.config = SimpleNamespace(
        max_steps=1,
        gradient_accumulation_steps=2,
        max_grad_norm=100.0,
        per_device_prompt_batch_size=2,
        logging_steps=1,
        save_steps=100,
        report_to=[],
    )
    trainer.output_dir = str(tmp_path)
    trainer.device = torch.device('cpu')
    trainer.accelerator = accelerator
    trainer.policy = SimpleNamespace(model=_FakePolicyModel(parameter))
    trainer.optimizer = torch.optim.SGD([parameter], lr=0.1)
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(
        trainer.optimizer,
        lr_lambda=lambda _: 1.0,
    )
    trainer.global_step = 0
    trainer.effective_world_size = 8
    trainer.prompts = list(range(128))
    trainer._prompt_cursor = trainer._prompt_cursor_after_steps(0)
    trainer.calibrate = lambda: None
    logged = []
    trainer._log = lambda metrics: logged.append(dict(metrics))

    def _run_micro_batch(accumulation_index):
        multiplier = 2.0 if accumulation_index == 0 else 4.0
        loss = parameter * multiplier
        return {
            'loss': loss,
            'metrics': {
                'loss': float(loss.detach().item()),
                'reward_score_sec_total': float(accumulation_index + 1),
                'diverse_minibatch/candidate_count': 4.0,
            },
            'step_peak_reserved': [0, 0],
            'step_peak_allocated': [0, 0],
        }

    trainer._run_training_micro_batch = _run_micro_batch
    trainer.train()

    assert accelerator.backward_losses == pytest.approx([2.0, 4.0])
    assert accelerator.no_sync_calls == 1
    assert parameter.item() == pytest.approx(0.7)
    assert trainer.global_step == 1
    assert trainer._prompt_cursor == 2
    assert len(logged) == 1
    assert logged[0]['loss'] == pytest.approx(3.0)
    assert logged[0]['reward_score_sec_total'] == pytest.approx(3.0)
    assert logged[0]['diverse_minibatch/candidate_count'] == pytest.approx(8.0)
    assert logged[0]['gradient_accumulation_steps'] == pytest.approx(2.0)
    assert logged[0]['effective_global_prompt_batch_size'] == pytest.approx(16.0)


def test_hbd_rejects_multi_micro_batch_memory_updates(tmp_path):
    config = _base_config()
    config.update(
        {
            'gradient_accumulation_steps': 2,
            'hbd': True,
        }
    )
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    with pytest.raises(
        ValueError,
        match='HBD does not support gradient_accumulation_steps > 1',
    ):
        load_config(config_path)
