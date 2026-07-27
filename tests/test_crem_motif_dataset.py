import csv
import importlib.util
import sqlite3
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_crem_motif_extension_dataset.py"
)
SPEC = importlib.util.spec_from_file_location("crem_motif_builder", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_test_fragments(path):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("name", "motif_extension"),
        )
        writer.writeheader()
        for index in range(10):
            writer.writerow(
                {
                    "name": f"test-{index}",
                    "motif_extension": f"[1*]CCCC{'C' * index}",
                }
            )


def test_iter_core_frequencies_supports_v0(tmp_path):
    path = tmp_path / "v0.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE radius3(core_smi TEXT, freq INTEGER)"
        )
        connection.executemany(
            "INSERT INTO radius3 VALUES (?, ?)",
            [("[*:1]CC", 2), ("[*:1]CC", 3), ("[*:1]CO", 7)],
        )
    with sqlite3.connect(path) as connection:
        rows = dict(
            MODULE._iter_core_frequencies(connection, set_name=None)
        )
    assert rows == {"[*:1]CC": 5, "[*:1]CO": 7}


def test_iter_core_frequencies_supports_v1(tmp_path):
    path = tmp_path / "v1.db"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version = 1")
        connection.execute(
            "CREATE TABLE frags(core_smi_id INTEGER, core_smi TEXT)"
        )
        connection.execute(
            """
            CREATE TABLE radius3(
                env_id INTEGER,
                core_smi_id INTEGER,
                core_num_atoms INTEGER,
                dist2 INTEGER,
                is_ring_closure INTEGER,
                chembl INTEGER
            )
            """
        )
        connection.executemany(
            "INSERT INTO frags VALUES (?, ?)",
            [(1, "[*:1]CC"), (2, "[*:1]CO")],
        )
        connection.executemany(
            "INSERT INTO radius3 VALUES (?, ?, ?, ?, ?, ?)",
            [(1, 1, 2, 0, 0, 2), (2, 1, 2, 0, 0, 3), (3, 2, 2, 0, 0, 7)],
        )
    with sqlite3.connect(path) as connection:
        rows = dict(
            MODULE._iter_core_frequencies(connection, set_name="chembl")
        )
    assert rows == {"[*:1]CC": 5, "[*:1]CO": 7}
