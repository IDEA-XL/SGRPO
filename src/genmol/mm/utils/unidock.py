from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import Thread

import numpy as np
import torch

from genmol.mm.crossdocked import _decode_atom_name
from genmol.mm.docking import (
    DEFAULT_FIXED_BOX_SIZE,
    _embed_ligand_from_smiles,
    _translate_ligand_to_center,
    _validate_written_ligand_sdf,
    _write_ligand_sdf,
)
from genmol.mm.utils.drugclip import _CrossDockedRawEntryStore


_UNIDOCK_ENERGY_PATTERN = re.compile(r'ENERGY=\s*([-+]?\d+(?:\.\d+)?)')
_AA_INDEX_TO_RESIDUE_NAME = {
    0: 'ALA',
    1: 'CYS',
    2: 'ASP',
    3: 'GLU',
    4: 'PHE',
    5: 'GLY',
    6: 'HIS',
    7: 'ILE',
    8: 'LYS',
    9: 'LEU',
    10: 'MET',
    11: 'ASN',
    12: 'PRO',
    13: 'GLN',
    14: 'ARG',
    15: 'SER',
    16: 'THR',
    17: 'VAL',
    18: 'TRP',
    19: 'TYR',
}
_HYDROGEN_ATOMIC_NUMBER = 1
_UNIDOCK_LIGAND_FAILURE_MARKERS = {
    'no_atoms': 'No atoms in this ligand.',
    'bond_length_assertion': 'model.cpp(1101)',
    'max_atom_count': 'exceed max atom counts.',
}


@dataclass(frozen=True)
class UniDockConfig:
    binary_path: str
    crossdocked_lmdb_path: str
    device: str
    batch_size: int = 128
    search_mode: str = 'fast'
    scoring: str = 'vina'
    num_modes: int = 1
    num_cpu: int = 0
    max_gpu_memory_mb: int = 0
    timeout_sec: int = 1800
    prepare_timeout_sec: int = 300
    box_size: tuple[float, float, float] = DEFAULT_FIXED_BOX_SIZE
    score_cache_size: int = 4096


class _UniDockCommandFailure(RuntimeError):
    def __init__(
        self,
        *,
        command: list[str],
        returncode: int,
        stdout: str,
        stderr: str,
        failure_kind: str | None,
    ) -> None:
        self.command = list(command)
        self.returncode = int(returncode)
        self.stdout = str(stdout)
        self.stderr = str(stderr)
        self.failure_kind = failure_kind
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        return (
            'Uni-Dock command failed:\n'
            f'command={" ".join(self.command)}\n'
            f'returncode={self.returncode}\n'
            f'stdout={self.stdout}\n'
            f'stderr={self.stderr}'
        )


def _stable_key(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def _ensure_positive_int(value, label: str) -> int:
    try:
        output = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{label} must be an integer, got {value!r}') from exc
    if output <= 0:
        raise ValueError(f'{label} must be positive, got {output}')
    return output


def _resolve_rank_cpu_count() -> int:
    raw_total_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if raw_total_cpus is None:
        total_cpus = os.cpu_count() or 1
    else:
        try:
            total_cpus = int(raw_total_cpus)
        except ValueError as exc:
            raise ValueError(f'SLURM_CPUS_PER_TASK must be an integer, got {raw_total_cpus!r}') from exc
    if total_cpus <= 0:
        raise ValueError(f'Invalid total CPU count for Uni-Dock: {total_cpus}')

    raw_local_world_size = os.environ.get('LOCAL_WORLD_SIZE') or os.environ.get('WORLD_SIZE') or '1'
    try:
        local_world_size = int(raw_local_world_size)
    except ValueError as exc:
        raise ValueError(
            f'LOCAL_WORLD_SIZE/WORLD_SIZE must be an integer, got {raw_local_world_size!r}'
        ) from exc
    if local_world_size <= 0:
        raise ValueError(f'Invalid local world size for Uni-Dock: {local_world_size}')
    return max(1, total_cpus // local_world_size)


def _resolve_visible_cuda_device(device: torch.device) -> str:
    raw_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    raw_local_rank = os.environ.get('LOCAL_RANK') or '0'
    try:
        local_rank = int(raw_local_rank)
    except ValueError as exc:
        raise ValueError(f'LOCAL_RANK must be an integer, got {raw_local_rank!r}') from exc
    if local_rank < 0:
        raise ValueError(f'LOCAL_RANK must be non-negative, got {local_rank}')

    if raw_visible_devices:
        visible_devices = [item.strip() for item in raw_visible_devices.split(',') if item.strip()]
        if not visible_devices:
            raise ValueError('CUDA_VISIBLE_DEVICES is set but empty')
        if len(visible_devices) == 1:
            return visible_devices[0]
        if local_rank >= len(visible_devices):
            raise ValueError(
                'LOCAL_RANK exceeds CUDA_VISIBLE_DEVICES entries for Uni-Dock: '
                f'{local_rank} vs {visible_devices}'
            )
        return visible_devices[local_rank]

    if device.type != 'cuda':
        raise ValueError(f'Uni-Dock requires a CUDA device, got {device}')
    if device.index is not None:
        return str(device.index)
    return str(local_rank)


def _resolve_native_ligand_center(raw_entry: dict) -> np.ndarray:
    try:
        from rdkit.Chem import GetPeriodicTable
    except ImportError as exc:
        raise RuntimeError('RDKit is required to compute Uni-Dock native-ligand centers') from exc

    required_keys = {'ligand_pos', 'ligand_element'}
    if not required_keys.issubset(raw_entry.keys()):
        missing = sorted(required_keys.difference(raw_entry.keys()))
        raise ValueError(f'Uni-Dock scoring requires native ligand fields for center alignment, missing {missing}')
    ligand_positions = np.asarray(raw_entry['ligand_pos'], dtype=np.float32)
    ligand_elements = np.asarray(raw_entry['ligand_element'], dtype=np.int64)
    if ligand_positions.ndim != 2 or ligand_positions.shape[1] != 3:
        raise ValueError(f'ligand_pos must have shape [num_atoms, 3], got {list(ligand_positions.shape)}')
    if ligand_elements.ndim != 1 or ligand_elements.shape[0] != ligand_positions.shape[0]:
        raise ValueError(
            'ligand_element must be length-aligned with ligand_pos for Uni-Dock center alignment'
        )
    if ligand_positions.shape[0] <= 0:
        raise ValueError('ligand_pos must contain at least one native ligand atom')
    periodic_table = GetPeriodicTable()
    masses = np.asarray(
        [float(periodic_table.GetAtomicWeight(int(atomic_number))) for atomic_number in ligand_elements.tolist()],
        dtype=np.float32,
    )
    if np.any(masses <= 0.0):
        raise ValueError(f'Encountered a non-positive native ligand atomic mass: {masses.tolist()}')
    total_mass = float(masses.sum())
    if total_mass <= 0.0:
        raise ValueError('Native ligand total mass must be positive for Uni-Dock center alignment')
    center = (ligand_positions * masses[:, None]).sum(axis=0) / total_mass
    return center.astype(np.float32)


def _prepare_ligand_sdf(smiles: str, center: np.ndarray, output_path: Path) -> None:
    ligand = _embed_ligand_from_smiles(smiles)
    ligand = _translate_ligand_to_center(ligand, center)
    _write_ligand_sdf(ligand, output_path)
    _validate_written_ligand_sdf(output_path, smiles=smiles)


def _prepare_ligand_sdf_task(
    smiles: str,
    center: np.ndarray,
    output_path: str,
    timeout_sec: int,
) -> tuple[bool, str | None]:
    normalized_timeout_sec = _ensure_positive_int(timeout_sec, 'unidock prepare_timeout_sec')
    result_holder = {
        'success': False,
        'failure_detail': None,
    }

    def _run_prepare():
        try:
            _prepare_ligand_sdf(smiles, center, Path(output_path))
        except Exception as exc:  # noqa: BLE001
            result_holder['failure_detail'] = f'{type(exc).__name__}: {exc}'
            return
        result_holder['success'] = True

    worker = Thread(
        target=_run_prepare,
        name='unidock_prepare_one',
        daemon=True,
    )
    worker.start()
    worker.join(timeout=normalized_timeout_sec)
    if worker.is_alive():
        return False, f'TimeoutError: ligand prepare exceeded {normalized_timeout_sec} seconds'
    return bool(result_holder['success']), result_holder['failure_detail']


def _is_prepare_timeout_failure(failure_detail: str | None) -> bool:
    return isinstance(failure_detail, str) and failure_detail.startswith('TimeoutError:')


def _parse_unidock_energy(output_path: Path) -> float:
    if not output_path.exists():
        raise FileNotFoundError(f'Uni-Dock output file not found: {output_path}')
    content = output_path.read_text()
    match = _UNIDOCK_ENERGY_PATTERN.search(content)
    if match is None:
        raise ValueError(f'Failed to parse Uni-Dock ENERGY from {output_path}')
    return float(match.group(1))


def _classify_unidock_failure(stdout: str, stderr: str) -> str | None:
    text = f'{stdout}\n{stderr}'
    for failure_kind, marker in _UNIDOCK_LIGAND_FAILURE_MARKERS.items():
        if marker in text:
            return failure_kind
    return None


def _periodic_table():
    try:
        from rdkit.Chem import GetPeriodicTable
    except ImportError as exc:
        raise RuntimeError('RDKit is required to normalize Uni-Dock receptor atoms') from exc
    return GetPeriodicTable()


def _element_symbol_from_atomic_number(atomic_number: int) -> str:
    try:
        normalized_atomic_number = int(atomic_number)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Invalid receptor atomic number: {atomic_number!r}') from exc
    if normalized_atomic_number <= 0:
        raise ValueError(f'Receptor atomic number must be positive, got {normalized_atomic_number}')
    symbol = str(_periodic_table().GetElementSymbol(normalized_atomic_number)).strip()
    if not symbol:
        raise ValueError(f'Failed to resolve element symbol for atomic number {normalized_atomic_number}')
    return symbol


def _first_alpha_index(text: str) -> int | None:
    for index, char in enumerate(text):
        if char.isalpha():
            return index
    return None


def _normalize_hydrogen_like_atom_name(atom_name: str) -> str:
    first_alpha_idx = _first_alpha_index(atom_name)
    if first_alpha_idx is None:
        raise ValueError(f'Hydrogen-like receptor atom name contains no alphabetic characters: {atom_name!r}')
    first_alpha = atom_name[first_alpha_idx].upper()
    if first_alpha == 'H':
        return atom_name
    if first_alpha == 'D':
        return f'{atom_name[:first_alpha_idx]}H{atom_name[first_alpha_idx + 1:]}'
    raise ValueError(
        'Hydrogen-like receptor atom name must begin with H or D at its first alphabetic position, '
        f'got {atom_name!r}'
    )


def _normalize_receptor_atom_name(atom_name: str, atomic_number: int | None = None) -> str:
    stripped = atom_name.strip()
    if not stripped:
        raise ValueError('Encountered an empty atom name while building a Uni-Dock receptor PDB')
    if len(stripped) > 4:
        raise ValueError(f'Atom name exceeds PDB width limit: {atom_name!r}')
    if atomic_number is not None and int(atomic_number) == _HYDROGEN_ATOMIC_NUMBER:
        stripped = _normalize_hydrogen_like_atom_name(stripped)
    return stripped


def _infer_pdb_element(atom_name: str) -> str:
    stripped = atom_name.strip()
    letters = ''.join(char for char in stripped if char.isalpha())
    if not letters:
        raise ValueError(f'Failed to infer element from atom name {atom_name!r}')
    return letters[0].upper()


def _format_pdb_atom_name(atom_name: str) -> str:
    return atom_name.strip().rjust(4)


def _resolve_receptor_atom_specs(raw_entry: dict) -> tuple[list[str], list[str]]:
    atom_names = [_decode_atom_name(item) for item in raw_entry['protein_atom_name']]
    raw_atomic_numbers = raw_entry.get('protein_element')
    if raw_atomic_numbers is None:
        normalized_atom_names = [_normalize_receptor_atom_name(atom_name) for atom_name in atom_names]
        element_symbols = [_infer_pdb_element(atom_name) for atom_name in normalized_atom_names]
        if any(symbol == 'D' for symbol in element_symbols):
            first_invalid_index = next(index for index, symbol in enumerate(element_symbols) if symbol == 'D')
            raise ValueError(
                'Uni-Dock receptor reconstruction requires protein_element when atom names imply deuterium-like '
                f'elements; first offending atom_name={normalized_atom_names[first_invalid_index]!r}'
            )
        return normalized_atom_names, element_symbols

    atomic_numbers = np.asarray(raw_atomic_numbers, dtype=np.int64)
    if atomic_numbers.ndim != 1:
        raise ValueError(f'protein_element must be a 1D array, got shape {list(atomic_numbers.shape)}')
    if atomic_numbers.shape[0] != len(atom_names):
        raise ValueError(
            'protein_element must be length-aligned with protein_atom_name for Uni-Dock receptor reconstruction: '
            f'{atomic_numbers.shape[0]} vs {len(atom_names)}'
        )

    normalized_atom_names: list[str] = []
    element_symbols: list[str] = []
    for atom_name, atomic_number in zip(atom_names, atomic_numbers.tolist()):
        normalized_atom_name = _normalize_receptor_atom_name(atom_name, atomic_number)
        element_symbol = _element_symbol_from_atomic_number(atomic_number)
        if element_symbol == 'D':
            raise ValueError(
                'Receptor element normalization unexpectedly produced D after atomic-number mapping: '
                f'atom_name={atom_name!r} atomic_number={atomic_number}'
            )
        normalized_atom_names.append(normalized_atom_name)
        element_symbols.append(element_symbol)
    return normalized_atom_names, element_symbols


def _build_receptor_pdb_lines(raw_entry: dict) -> list[str]:
    required_keys = {'protein_pos', 'protein_atom_name', 'protein_atom_to_aa_type'}
    if not required_keys.issubset(raw_entry.keys()):
        missing = sorted(required_keys.difference(raw_entry.keys()))
        raise ValueError(f'Uni-Dock receptor reconstruction is missing required keys: {missing}')

    atom_positions = np.asarray(raw_entry['protein_pos'], dtype=np.float32)
    atom_names, element_symbols = _resolve_receptor_atom_specs(raw_entry)
    atom_to_aa_type = np.asarray(raw_entry['protein_atom_to_aa_type'], dtype=np.int64)
    if atom_positions.ndim != 2 or atom_positions.shape[1] != 3:
        raise ValueError(f'protein_pos must have shape [num_atoms, 3], got {list(atom_positions.shape)}')
    if len(atom_names) != atom_positions.shape[0] or atom_to_aa_type.shape[0] != atom_positions.shape[0]:
        raise ValueError('protein atom-level arrays must share the same length for Uni-Dock receptor reconstruction')
    if len(element_symbols) != atom_positions.shape[0]:
        raise ValueError('Resolved receptor element symbols must share the same length as protein_pos')

    residue_start_indices = [idx for idx, atom_name in enumerate(atom_names) if atom_name == 'N']
    if not residue_start_indices:
        raise ValueError('Failed to reconstruct Uni-Dock receptor residues: no backbone N atoms found')

    residue_start_indices.append(len(atom_names))
    pdb_lines: list[str] = []
    atom_serial = 1
    for residue_serial, (start_idx, end_idx) in enumerate(zip(residue_start_indices[:-1], residue_start_indices[1:]), start=1):
        residue_atom_names = atom_names[start_idx:end_idx]
        residue_atom_types = atom_to_aa_type[start_idx:end_idx]
        residue_positions = atom_positions[start_idx:end_idx]
        unique_types = np.unique(residue_atom_types)
        if unique_types.size != 1:
            raise ValueError(
                'Failed to reconstruct Uni-Dock receptor residues deterministically: '
                f'multiple amino acid types in one residue segment {unique_types.tolist()}'
            )
        residue_name = _AA_INDEX_TO_RESIDUE_NAME.get(int(unique_types[0]))
        if residue_name is None:
            raise ValueError(f'Unsupported amino acid index for Uni-Dock receptor reconstruction: {int(unique_types[0])}')
        residue_element_symbols = element_symbols[start_idx:end_idx]
        for atom_name, atom_position, element_symbol in zip(residue_atom_names, residue_positions, residue_element_symbols):
            x, y, z = (float(atom_position[0]), float(atom_position[1]), float(atom_position[2]))
            pdb_lines.append(
                f"ATOM  {atom_serial:5d} {_format_pdb_atom_name(atom_name)} {residue_name:>3s} A{residue_serial:4d}"
                f"    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element_symbol:>2s}\n"
            )
            atom_serial += 1
    if not pdb_lines:
        raise ValueError('Failed to reconstruct any Uni-Dock receptor atoms')
    pdb_lines.append('END\n')
    return pdb_lines


class UniDockScorer:
    def __init__(self, config: UniDockConfig):
        self.config = config
        self.binary_path = shutil.which(config.binary_path) if os.path.sep not in config.binary_path else config.binary_path
        if self.binary_path is None:
            raise RuntimeError(f'Uni-Dock binary not found: {config.binary_path}')
        self.binary_path = str(Path(self.binary_path).expanduser().resolve())
        if not os.access(self.binary_path, os.X_OK):
            raise PermissionError(f'Uni-Dock binary is not executable: {self.binary_path}')
        self.crossdocked_lmdb_path = str(Path(config.crossdocked_lmdb_path).expanduser().resolve())
        if not Path(self.crossdocked_lmdb_path).exists():
            raise FileNotFoundError(f'Uni-Dock CrossDocked LMDB not found: {self.crossdocked_lmdb_path}')
        self.device = torch.device(config.device)
        if self.device.type != 'cuda':
            raise ValueError(f'Uni-Dock requires device=cuda, got {self.device}')
        self.batch_size = _ensure_positive_int(config.batch_size, 'unidock batch_size')
        self.num_modes = _ensure_positive_int(config.num_modes, 'unidock num_modes')
        self.timeout_sec = _ensure_positive_int(config.timeout_sec, 'unidock timeout_sec')
        self.prepare_timeout_sec = _ensure_positive_int(
            config.prepare_timeout_sec,
            'unidock prepare_timeout_sec',
        )
        if int(config.score_cache_size) <= 0:
            raise ValueError(f'unidock score_cache_size must be positive, got {config.score_cache_size}')
        self.max_gpu_memory_mb = int(config.max_gpu_memory_mb)
        if self.max_gpu_memory_mb < 0:
            raise ValueError(f'unidock max_gpu_memory_mb must be non-negative, got {self.max_gpu_memory_mb}')
        if str(config.scoring) != 'vina':
            raise ValueError(f"Uni-Dock reward currently supports only scoring='vina', got {config.scoring!r}")
        self.rank_cpu_count = _resolve_rank_cpu_count() if int(config.num_cpu) <= 0 else _ensure_positive_int(
            config.num_cpu, 'unidock num_cpu'
        )
        self.visible_cuda_device = _resolve_visible_cuda_device(self.device)
        self.box_size = np.asarray(config.box_size, dtype=np.float32)
        if self.box_size.shape != (3,):
            raise ValueError(f'unidock box_size must be length-3, got shape {self.box_size.shape}')
        if np.any(self.box_size <= 0.0):
            raise ValueError(f'unidock box_size must be strictly positive, got {self.box_size.tolist()}')
        self._raw_entry_store = _CrossDockedRawEntryStore(self.crossdocked_lmdb_path)
        self._receptor_cache_dir = Path(tempfile.mkdtemp(prefix='unidock_receptor_cache_'))
        self._receptor_path_cache: dict[str, Path] = {}
        self._score_cache: OrderedDict[tuple[str, str], float] = OrderedDict()
        self._prepare_executor = None
        if self.rank_cpu_count > 1:
            self._prepare_executor = ThreadPoolExecutor(
                max_workers=self.rank_cpu_count,
                thread_name_prefix='unidock_prepare',
            )
        self.last_score_stats = self._empty_stats()

    @staticmethod
    def _empty_stats():
        return {
            'unidock_input_count': 0,
            'unidock_unique_smiles_count': 0,
            'unidock_unique_pocket_count': 0,
            'unidock_score_success_count': 0,
            'unidock_score_failure_count': 0,
            'unidock_score_success_fraction': 0.0,
            'unidock_cache_hit_count': 0,
            'unidock_cache_miss_count': 0,
            'unidock_prepare_sec': 0.0,
            'unidock_dock_sec': 0.0,
            'unidock_parse_sec': 0.0,
            'unidock_chunk_count': 0,
            'unidock_rank_cpu_count': 0,
            'unidock_prepare_worker_count': 0,
            'unidock_max_gpu_memory_mb': 0,
            'unidock_fail_prepare_count': 0,
            'unidock_fail_prepare_timeout_count': 0,
            'unidock_fail_output_missing_count': 0,
            'unidock_fail_parse_count': 0,
            'unidock_fail_runtime_count': 0,
            'unidock_fail_isolated_ligand_count': 0,
            'unidock_runtime_split_count': 0,
        }

    def close(self):
        if self._prepare_executor is not None:
            self._prepare_executor.shutdown(wait=True, cancel_futures=False)
        self._raw_entry_store.close()
        shutil.rmtree(self._receptor_cache_dir, ignore_errors=True)

    def _cache_get(self, cache_key: tuple[str, str]) -> float | None:
        cached = self._score_cache.get(cache_key)
        if cached is None:
            return None
        self._score_cache.move_to_end(cache_key)
        return cached

    def _cache_put(self, cache_key: tuple[str, str], value: float) -> None:
        self._score_cache[cache_key] = float(value)
        self._score_cache.move_to_end(cache_key)
        while len(self._score_cache) > int(self.config.score_cache_size):
            self._score_cache.popitem(last=False)

    def _build_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env['CUDA_VISIBLE_DEVICES'] = self.visible_cuda_device
        env['OMP_NUM_THREADS'] = str(self.rank_cpu_count)
        return env

    @staticmethod
    def _pocket_cache_key(pocket_entry: dict) -> str:
        if 'source_index' in pocket_entry:
            return f"source_{int(pocket_entry['source_index'])}"
        if 'lmdb_key' in pocket_entry:
            return f"lmdb_{int(pocket_entry['lmdb_key'])}"
        raise ValueError('Uni-Dock scoring requires source_index or lmdb_key in pocket entries')

    def _resolve_receptor_pdb_path(self, pocket_entry: dict) -> Path:
        cache_key = self._pocket_cache_key(pocket_entry)
        cached = self._receptor_path_cache.get(cache_key)
        if cached is not None:
            return cached
        raw_entry = self._raw_entry_store.get_entry(pocket_entry)
        receptor_path = self._receptor_cache_dir / f'{cache_key}.pdb'
        receptor_path.write_text(''.join(_build_receptor_pdb_lines(raw_entry)))
        self._receptor_path_cache[cache_key] = receptor_path
        return receptor_path

    def _execute_unidock_command(
        self,
        *,
        receptor_path: Path,
        pocket_center: np.ndarray,
        ligand_paths: list[Path],
        output_dir: Path,
    ) -> None:
        command = [
            self.binary_path,
            '--receptor',
            str(receptor_path),
            '--gpu_batch',
            *[str(ligand_path) for ligand_path in ligand_paths],
            '--dir',
            str(output_dir),
            '--search_mode',
            str(self.config.search_mode),
            '--scoring',
            str(self.config.scoring),
            '--center_x',
            str(float(pocket_center[0])),
            '--center_y',
            str(float(pocket_center[1])),
            '--center_z',
            str(float(pocket_center[2])),
            '--size_x',
            str(float(self.box_size[0])),
            '--size_y',
            str(float(self.box_size[1])),
            '--size_z',
            str(float(self.box_size[2])),
            '--num_modes',
            str(self.num_modes),
            '--cpu',
            str(self.rank_cpu_count),
            '--device_id',
            '0',
            '--verbosity',
            '0',
        ]
        if self.max_gpu_memory_mb > 0:
            command.extend(['--max_gpu_memory', str(self.max_gpu_memory_mb)])

        dock_start = time.perf_counter()
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=self.timeout_sec,
            env=self._build_env(),
        )
        self.last_score_stats['unidock_dock_sec'] += time.perf_counter() - dock_start
        if result.returncode != 0:
            raise _UniDockCommandFailure(
                command=command,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                failure_kind=_classify_unidock_failure(result.stdout, result.stderr),
            )

    def _parse_unidock_outputs(
        self,
        *,
        output_dir: Path,
        ligand_input_paths: list[Path],
        prepared_indices: list[int],
        outputs: list[float | None],
    ) -> None:
        parse_start = time.perf_counter()
        for prepared_idx in prepared_indices:
            ligand_path = ligand_input_paths[prepared_idx]
            output_path = output_dir / f'{ligand_path.stem}_out.sdf'
            if not output_path.exists():
                matches = sorted(output_dir.glob(f'{ligand_path.stem}*.sdf'))
                if len(matches) != 1:
                    self.last_score_stats['unidock_fail_output_missing_count'] += 1
                    continue
                output_path = matches[0]
            try:
                outputs[prepared_idx] = _parse_unidock_energy(output_path)
            except Exception:
                self.last_score_stats['unidock_fail_parse_count'] += 1
                continue
        self.last_score_stats['unidock_parse_sec'] += time.perf_counter() - parse_start

    def _score_prepared_indices(
        self,
        *,
        receptor_path: Path,
        pocket_center: np.ndarray,
        ligand_input_paths: list[Path],
        prepared_indices: list[int],
        outputs: list[float | None],
        output_root_dir: Path,
    ) -> None:
        if not prepared_indices:
            return
        self.last_score_stats['unidock_chunk_count'] += 1
        output_dir = output_root_dir / f'run_{self.last_score_stats["unidock_chunk_count"]:04d}'
        output_dir.mkdir(parents=True, exist_ok=True)
        ligand_paths = [ligand_input_paths[prepared_idx] for prepared_idx in prepared_indices]
        try:
            self._execute_unidock_command(
                receptor_path=receptor_path,
                pocket_center=pocket_center,
                ligand_paths=ligand_paths,
                output_dir=output_dir,
            )
        except _UniDockCommandFailure as exc:
            if exc.failure_kind is None:
                raise RuntimeError(str(exc)) from exc
            self.last_score_stats['unidock_fail_runtime_count'] += 1
            if len(prepared_indices) == 1:
                self.last_score_stats['unidock_fail_isolated_ligand_count'] += 1
                outputs[prepared_indices[0]] = None
                return
            midpoint = len(prepared_indices) // 2
            if midpoint <= 0:
                raise RuntimeError(str(exc)) from exc
            self.last_score_stats['unidock_runtime_split_count'] += 1
            self._score_prepared_indices(
                receptor_path=receptor_path,
                pocket_center=pocket_center,
                ligand_input_paths=ligand_input_paths,
                prepared_indices=prepared_indices[:midpoint],
                outputs=outputs,
                output_root_dir=output_root_dir,
            )
            self._score_prepared_indices(
                receptor_path=receptor_path,
                pocket_center=pocket_center,
                ligand_input_paths=ligand_input_paths,
                prepared_indices=prepared_indices[midpoint:],
                outputs=outputs,
                output_root_dir=output_root_dir,
            )
            return
        self._parse_unidock_outputs(
            output_dir=output_dir,
            ligand_input_paths=ligand_input_paths,
            prepared_indices=prepared_indices,
            outputs=outputs,
        )

    def _run_unidock_chunk(
        self,
        receptor_path: Path,
        pocket_center: np.ndarray,
        smiles_chunk: list[str],
        pocket_key: str,
    ) -> list[float | None]:
        if not smiles_chunk:
            return []
        with tempfile.TemporaryDirectory(prefix=f'unidock_{pocket_key}_') as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            ligand_dir = temp_dir / 'ligands'
            output_root_dir = temp_dir / 'outputs'
            ligand_dir.mkdir(parents=True, exist_ok=True)
            output_root_dir.mkdir(parents=True, exist_ok=True)

            prepare_start = time.perf_counter()
            ligand_input_paths: list[Path] = []
            prepared_indices: list[int] = []
            self.last_score_stats['unidock_prepare_worker_count'] = int(
                max(self.last_score_stats['unidock_prepare_worker_count'], min(self.rank_cpu_count, len(smiles_chunk)))
            )
            if self._prepare_executor is None or len(smiles_chunk) == 1:
                for ligand_idx, smiles in enumerate(smiles_chunk):
                    ligand_path = ligand_dir / f'lig_{ligand_idx:04d}_{_stable_key(smiles)}.sdf'
                    success, failure_detail = _prepare_ligand_sdf_task(
                        smiles,
                        pocket_center.copy(),
                        str(ligand_path),
                        self.prepare_timeout_sec,
                    )
                    if not success:
                        self.last_score_stats['unidock_fail_prepare_count'] += 1
                        if _is_prepare_timeout_failure(failure_detail):
                            self.last_score_stats['unidock_fail_prepare_timeout_count'] += 1
                        ligand_input_paths.append(ligand_path)
                        continue
                    ligand_input_paths.append(ligand_path)
                    prepared_indices.append(ligand_idx)
            else:
                future_to_index = {}
                for ligand_idx, smiles in enumerate(smiles_chunk):
                    ligand_path = ligand_dir / f'lig_{ligand_idx:04d}_{_stable_key(smiles)}.sdf'
                    ligand_input_paths.append(ligand_path)
                    future = self._prepare_executor.submit(
                        _prepare_ligand_sdf_task,
                        smiles,
                        pocket_center.copy(),
                        str(ligand_path),
                        self.prepare_timeout_sec,
                    )
                    future_to_index[future] = ligand_idx
                for future in as_completed(future_to_index):
                    ligand_idx = future_to_index[future]
                    success, failure_detail = future.result()
                    if not success:
                        self.last_score_stats['unidock_fail_prepare_count'] += 1
                        if _is_prepare_timeout_failure(failure_detail):
                            self.last_score_stats['unidock_fail_prepare_timeout_count'] += 1
                        continue
                    prepared_indices.append(ligand_idx)
                prepared_indices.sort()
            self.last_score_stats['unidock_prepare_sec'] += time.perf_counter() - prepare_start

            outputs: list[float | None] = [None for _ in smiles_chunk]
            if not prepared_indices:
                return outputs
            self._score_prepared_indices(
                receptor_path=receptor_path,
                pocket_center=pocket_center,
                ligand_input_paths=ligand_input_paths,
                prepared_indices=prepared_indices,
                outputs=outputs,
                output_root_dir=output_root_dir,
            )
            return outputs

    def score(self, smiles_list: list[str], pocket_entries: list[dict]) -> list[float | None]:
        if len(smiles_list) != len(pocket_entries):
            raise ValueError(
                'Uni-Dock scoring requires smiles_list and pocket_entries to have the same length: '
                f'{len(smiles_list)} vs {len(pocket_entries)}'
            )
        self.last_score_stats = self._empty_stats()
        self.last_score_stats['unidock_input_count'] = int(len(smiles_list))
        self.last_score_stats['unidock_rank_cpu_count'] = int(self.rank_cpu_count)
        self.last_score_stats['unidock_prepare_worker_count'] = 1 if self._prepare_executor is None else int(self.rank_cpu_count)
        self.last_score_stats['unidock_max_gpu_memory_mb'] = int(self.max_gpu_memory_mb)
        if not smiles_list:
            return []

        grouped_indices: OrderedDict[str, list[int]] = OrderedDict()
        pocket_payloads: dict[str, tuple[Path, np.ndarray]] = {}
        unique_smiles_keys: OrderedDict[tuple[str, str], None] = OrderedDict()
        outputs: list[float | None] = [None for _ in smiles_list]

        for index, (smiles, pocket_entry) in enumerate(zip(smiles_list, pocket_entries)):
            pocket_key = self._pocket_cache_key(pocket_entry)
            grouped_indices.setdefault(pocket_key, []).append(index)
            if pocket_key not in pocket_payloads:
                raw_entry = self._raw_entry_store.get_entry(pocket_entry)
                pocket_payloads[pocket_key] = (
                    self._resolve_receptor_pdb_path(pocket_entry),
                    _resolve_native_ligand_center(raw_entry),
                )
            unique_smiles_keys[(pocket_key, smiles)] = None

        self.last_score_stats['unidock_unique_pocket_count'] = int(len(grouped_indices))
        self.last_score_stats['unidock_unique_smiles_count'] = int(len(unique_smiles_keys))

        for pocket_key, indices in grouped_indices.items():
            receptor_path, pocket_center = pocket_payloads[pocket_key]
            unique_smiles: list[str] = []
            unique_cache_keys: list[tuple[str, str]] = []
            cache_key_to_indices: dict[tuple[str, str], list[int]] = {}
            cached_scores: dict[tuple[str, str], float] = {}
            for index in indices:
                smiles = smiles_list[index]
                cache_key = (pocket_key, smiles)
                cache_key_to_indices.setdefault(cache_key, []).append(index)
                if cache_key_to_indices[cache_key][0] != index:
                    continue
                cached = self._cache_get(cache_key)
                if cached is not None:
                    self.last_score_stats['unidock_cache_hit_count'] += 1
                    cached_scores[cache_key] = float(cached)
                    continue
                self.last_score_stats['unidock_cache_miss_count'] += 1
                unique_smiles.append(smiles)
                unique_cache_keys.append(cache_key)

            for cache_key, cached_score in cached_scores.items():
                for index in cache_key_to_indices[cache_key]:
                    outputs[index] = float(cached_score)

            if not unique_smiles:
                continue

            for chunk_start in range(0, len(unique_smiles), self.batch_size):
                smiles_chunk = unique_smiles[chunk_start:chunk_start + self.batch_size]
                cache_key_chunk = unique_cache_keys[chunk_start:chunk_start + self.batch_size]
                chunk_scores = self._run_unidock_chunk(
                    receptor_path=receptor_path,
                    pocket_center=pocket_center,
                    smiles_chunk=smiles_chunk,
                    pocket_key=pocket_key,
                )
                if len(chunk_scores) != len(smiles_chunk):
                    raise RuntimeError(
                        'Uni-Dock scorer returned mismatched score count: '
                        f'expected {len(smiles_chunk)}, got {len(chunk_scores)}'
                    )
                for cache_key, score in zip(cache_key_chunk, chunk_scores):
                    if score is None:
                        continue
                    self._cache_put(cache_key, float(score))
                    for index in cache_key_to_indices[cache_key]:
                        outputs[index] = float(score)

        success_count = sum(score is not None for score in outputs)
        self.last_score_stats['unidock_score_success_count'] = int(success_count)
        self.last_score_stats['unidock_score_failure_count'] = int(len(outputs) - success_count)
        if outputs:
            self.last_score_stats['unidock_score_success_fraction'] = float(success_count / len(outputs))
        return outputs
