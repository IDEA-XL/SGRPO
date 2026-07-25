import argparse
import json
from pathlib import Path

from rl_shared.diverse_minibatch import (
    select_molecule_groups,
    select_sequence_groups,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Benchmark exact diverse mini-batch selection on JSONL data.',
    )
    parser.add_argument('--input-jsonl', required=True)
    parser.add_argument('--item-field', required=True)
    parser.add_argument(
        '--kind',
        required=True,
        choices=('molecule', 'sequence'),
    )
    parser.add_argument('--candidate-size', required=True, type=int)
    parser.add_argument('--selected-size', required=True, type=int)
    parser.add_argument('--seed', default=0, type=int)
    return parser.parse_args()


def load_items(path, item_field, candidate_size):
    if candidate_size <= 0:
        raise ValueError('candidate_size must be positive')
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f'input JSONL not found: {input_path}')

    items = []
    with input_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(
                    f'JSONL row {line_number} must be an object'
                )
            item = row.get(item_field)
            if item is None or str(item).strip() == '':
                continue
            items.append(item)
            if len(items) == candidate_size:
                return items
    raise ValueError(
        f'input contains only {len(items)} non-empty {item_field!r} values; '
        f'{candidate_size} are required'
    )


def main():
    args = parse_args()
    items = load_items(
        args.input_jsonl,
        args.item_field,
        args.candidate_size,
    )
    if args.kind == 'molecule':
        result = select_molecule_groups(
            items,
            candidate_size=args.candidate_size,
            selected_size=args.selected_size,
            seed=args.seed,
        )
    else:
        result = select_sequence_groups(
            items,
            candidate_size=args.candidate_size,
            selected_size=args.selected_size,
            seed=args.seed,
        )
    payload = {
        'kind': args.kind,
        'input_jsonl': str(Path(args.input_jsonl).resolve()),
        'item_field': args.item_field,
        'candidate_size': args.candidate_size,
        'selected_size': args.selected_size,
        'seed': args.seed,
        'metrics': result.metrics,
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == '__main__':
    main()
