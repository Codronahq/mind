"""Gate the boundary, not a mock of it.

Every fixture here writes a real parquet and a real sidecar manifest, because
the failure this module guards is a file on disk disagreeing with a file beside
it. A fixture that patched the reader would pass against a reader that never
opened anything.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from codrona_mind import responses

# (user, problem, submitted_at, in_public_problemset)
Row = tuple[str, str, str, bool]

ROWS: list[Row] = [
    # Trains: both sides seen before the cutoff.
    ("u1", "100A", "2024-03-01", True),
    ("u2", "100A", "2024-05-01", True),
    ("u1", "200B", "2025-02-01", True),
    # Held out ON THE CUTOFF INSTANT. Train is `< cutoff` and test is
    # `>= cutoff`, so this belongs to test; widening train to `<=` would move
    # it and nothing else in this file sits on a boundary to notice.
    ("u1", "100A", "2026-01-01", True),
    # Held out: u1 and 100A are both known -> G1.
    ("u1", "100A", "2026-02-01", True),
    # Held out: u1 known, 900Z unseen -> G3.
    ("u1", "900Z", "2026-03-01", True),
    # Held out: u9 unseen, 100A known -> G2.
    ("u9", "100A", "2026-04-01", True),
    # Held out: neither seen -> both new.
    ("u8", "800Y", "2026-05-01", True),
    # Outside the bank entirely: must never reach the partition.
    ("u1", "700G", "2026-06-01", False),
]


def _write(path: pathlib.Path, rows: list[Row] | None = None) -> pa.Table:
    data = ROWS if rows is None else rows
    table = pa.table(
        {
            "user_key": pa.array([r[0] for r in data], pa.string()),
            "problem_key": pa.array([r[1] for r in data], pa.string()),
            "submitted_at": pa.array(
                [dt.datetime.fromisoformat(r[2]) for r in data],
                pa.timestamp("us"),
            ),
            "in_public_problemset": pa.array([r[3] for r in data], pa.bool_()),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return table


def _manifest(path: pathlib.Path, table: pa.Table, **overrides: Any) -> dict[str, Any]:
    """A sidecar shaped exactly as lens writes one, with the split lens measured."""
    payload: dict[str, Any] = {
        "artefact": "responses.parquet",
        "schema": [{"name": name, "type": "IGNORED"} for name in table.schema.names],
        "counts": {"merged_responses": table.num_rows},
        "split": {
            "cutoff": responses.DEFAULT_CUTOFF,
            "train": 3,
            "test": 5,
            "g1_known_user_known_item": 2,
            "g3_new_item_only": 1,
            "g2_new_user_only": 1,
            "both_new": 1,
        },
    }
    for key, value in overrides.items():
        section, _, field = key.partition("__")
        if field:
            payload[section][field] = value
        else:
            payload[section] = value
    responses.sidecar_manifest(path).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def _artefact(tmp_path: pathlib.Path, **overrides: Any) -> pathlib.Path:
    path = tmp_path / "responses.parquet"
    table = _write(path)
    _manifest(path, table, **overrides)
    return path


def test_the_sidecar_path_matches_the_one_lens_writes(tmp_path: pathlib.Path) -> None:
    """Both repos construct this name independently; they must agree exactly."""
    artefact = tmp_path / "responses.parquet"
    assert responses.sidecar_manifest(artefact).name == "responses.manifest.json"
    assert responses.sidecar_manifest(artefact).parent == tmp_path


def test_a_missing_sidecar_raises_rather_than_defaulting(
    tmp_path: pathlib.Path,
) -> None:
    """No manifest means nothing can be verified, which is not a pass."""
    path = tmp_path / "responses.parquet"
    _write(path)
    with pytest.raises(FileNotFoundError, match="no sidecar manifest"):
        responses.load_manifest(path)


def test_shape_comes_from_the_footer(tmp_path: pathlib.Path) -> None:
    artefact = _artefact(tmp_path)
    rows, columns = responses.artefact_shape(artefact)
    assert rows == len(ROWS)
    assert columns == ["user_key", "problem_key", "submitted_at", "in_public_problemset"]


def test_a_matching_artefact_reports_nothing(tmp_path: pathlib.Path) -> None:
    artefact = _artefact(tmp_path)
    assert responses.verify_artefact(artefact, responses.load_manifest(artefact)) == []


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"counts": {"merged_responses": 99}}, "artefact rows"),
        ({"schema": [{"name": "user_key", "type": "X"}]}, "absent from manifest"),
        (
            {
                "schema": [
                    {"name": n, "type": "X"}
                    for n in ["problem_key", "user_key", "submitted_at", "in_public_problemset"]
                ]
            },
            "column ORDER differs",
        ),
        ({"schema": []}, "no schema block"),
    ],
)
def test_every_shape_disagreement_is_reported(
    tmp_path: pathlib.Path, overrides: dict[str, Any], expected: str
) -> None:
    artefact = _artefact(tmp_path, **overrides)
    problems = responses.verify_artefact(artefact, responses.load_manifest(artefact))
    assert any(expected in line for line in problems), problems


def test_a_renamed_column_is_reported(tmp_path: pathlib.Path) -> None:
    """Names are compared even though types are not."""
    artefact = _artefact(
        tmp_path,
        schema=[
            {"name": n, "type": "X"} for n in ["user_key", "problem_key", "submitted_at", "in_bank"]
        ],
    )
    problems = responses.verify_artefact(artefact, responses.load_manifest(artefact))
    assert any("in_bank" in line for line in problems), problems
    assert any("in_public_problemset" in line for line in problems), problems


def test_the_bank_filter_drops_out_of_bank_rows(tmp_path: pathlib.Path) -> None:
    artefact = _artefact(tmp_path)
    bank = responses.load_bank(artefact, responses.SPLIT_COLUMNS)
    assert bank.num_rows == len(ROWS) - 1
    assert "700G" not in bank.column("problem_key").to_pylist()


def test_the_partition_classifies_each_held_out_response(
    tmp_path: pathlib.Path,
) -> None:
    """One row per class, so no class passes by being empty."""
    artefact = _artefact(tmp_path)
    bank = responses.load_bank(artefact, responses.SPLIT_COLUMNS)
    part = responses.partition(bank)
    assert part.train == 3
    assert part.test == 5
    assert part.g1_known_user_known_item == 2
    assert part.g3_new_item_only == 1
    assert part.g2_new_user_only == 1
    assert part.both_new == 1
    assert part.parts == part.test


def test_the_partition_agrees_with_the_manifest(tmp_path: pathlib.Path) -> None:
    artefact = _artefact(tmp_path)
    manifest = responses.load_manifest(artefact)
    bank = responses.load_bank(artefact, responses.SPLIT_COLUMNS)
    assert responses.verify_partition(responses.partition(bank), manifest) == []


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("train", "split.train"),
        ("test", "split.test"),
        ("g1_known_user_known_item", "split.g1_known_user_known_item"),
        ("both_new", "split.both_new"),
    ],
)
def test_a_split_disagreement_stops_the_build(
    tmp_path: pathlib.Path, field: str, expected: str
) -> None:
    """The point of recomputing: lens measured in DuckDB, this measures in Arrow."""
    artefact = _artefact(tmp_path)
    manifest = responses.load_manifest(artefact)
    manifest["split"][field] += 1
    bank = responses.load_bank(artefact, responses.SPLIT_COLUMNS)
    problems = responses.verify_partition(responses.partition(bank), manifest)
    assert any(expected in line for line in problems), problems


def test_a_different_cutoff_is_refused_rather_than_compared(
    tmp_path: pathlib.Path,
) -> None:
    """Counts under two cutoffs describe different splits and must not be diffed."""
    artefact = _artefact(tmp_path)
    manifest = responses.load_manifest(artefact)
    bank = responses.load_bank(artefact, responses.SPLIT_COLUMNS)
    problems = responses.verify_partition(responses.partition(bank, "2025-01-01"), manifest)
    assert len(problems) == 1
    assert "does not match the manifest" in problems[0]


def test_a_cutoff_after_everything_leaves_an_empty_held_out_period(
    tmp_path: pathlib.Path,
) -> None:
    artefact = _artefact(tmp_path)
    bank = responses.load_bank(artefact, responses.SPLIT_COLUMNS)
    part = responses.partition(bank, "2099-01-01")
    assert part.test == 0
    assert part.parts == 0
    assert part.train == bank.num_rows


def test_a_partition_whose_parts_do_not_sum_is_reported() -> None:
    """Built by hand, because a correct `partition` can never produce this.

    The four classes come from mutually exclusive masks, so the check is inert
    against the real function and can only fire if that function breaks. Testing
    it needs a Partition constructed directly - the same shape lens uses for its
    equivalent invariant.
    """
    part = responses.Partition(
        cutoff=responses.DEFAULT_CUTOFF,
        train=3,
        test=5,
        g1_known_user_known_item=1,
        g3_new_item_only=1,
        g2_new_user_only=1,
        both_new=1,
    )
    manifest = {
        "split": {
            "cutoff": responses.DEFAULT_CUTOFF,
            "train": 3,
            "test": 5,
            "g1_known_user_known_item": 1,
            "g3_new_item_only": 1,
            "g2_new_user_only": 1,
            "both_new": 1,
        }
    }
    problems = responses.verify_partition(part, manifest)
    assert len(problems) == 1
    assert "parts sum to 4 against 5" in problems[0]


def test_the_cli_verifies_end_to_end(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artefact = _artefact(tmp_path)
    assert responses.main(["--artefact", str(artefact), "--verify"]) == 0
    assert "agrees with the manifest" in capsys.readouterr().out


def test_the_cli_fails_on_a_split_disagreement(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artefact = _artefact(tmp_path, split__both_new=99)
    assert responses.main(["--artefact", str(artefact), "--verify"]) == 1
    assert "split.both_new" in capsys.readouterr().err


def test_the_cli_skips_loudly_without_an_artefact(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A machine that never built one is not a failing machine."""
    assert responses.main(["--artefact", str(tmp_path / "gone.parquet")]) == 0
    assert "nothing verified" in capsys.readouterr().out
