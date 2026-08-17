"""Read the response matrix `lens` builds, and refuse to fit a stale one.

THIS MODULE IS THE REPO BOUNDARY MADE REAL. `codrona.md` §14 settles that
warehouse-reading code lives in `lens` and every other repo consumes an emitted
file: `lens` writes, the consumer reads a file and never sees DuckDB. Nothing in
here opens a warehouse, and nothing in here recomputes anything `lens` already
measured - it reads what `lens` wrote and checks that it is what `lens` says it
is.

THE FAILURE THIS EXISTS TO PREVENT. `lens` gates its artefact against
`exports/model/responses.manifest.json`, which is committed in `lens` and which
this repo has no access to. Without a copy crossing the boundary, a fit here
could run against a parquet rebuilt from a moved warehouse, or half-written, or
produced by an older code path, and every test in both repos would stay green.
So `lens` writes a sidecar manifest beside the artefact and this module reads
it, which makes the artefact directory self-describing.

REFUSAL IS ON THE READ PATH, NOT ONLY IN THE COMMAND. Until 17 Aug 2026 the
first line of this docstring claimed a refusal the module did not implement.
`verify_artefact` and `verify_partition` returned lists of strings, `main`
turned a non-empty list into an exit code, and `load_bank` - the path a fit
actually takes - read the parquet without opening the manifest at all. A
verification you have to remember to run is not a gate, which is precisely the
defect `compare_manifest` carried in `lens` one repo over. `load_bank` now
verifies by default and raises `StaleArtefactError`. The check is a footer read
and a JSON parse, so there is no cost worth saving by skipping it.

WHAT IS CHECKED, AND WHAT IS NOT. Row count and column names and order come from
the parquet footer, so verification is metadata-only and costs nothing. Column
TYPES are deliberately not compared: the manifest records DuckDB type names
(`VARCHAR`, `BIGINT`, `TIMESTAMP`) and Arrow reports its own (`string`,
`int64`, `timestamp[us]`), so a mapping between them would be a table of
guesses that fails on the first type either engine adds. Types are compared
inside `lens`, where both sides come from the same engine. Naming what a gate
does not cover is the point of stating it here rather than implying coverage.

THE PARTITION IS RECOMPUTED ON PURPOSE. `verify_partition` measures the
evaluation split here, from the artefact, and compares it against the counts
`lens` recorded from the warehouse. Two engines, two code paths, one answer -
the same pairing that confirmed every corpus figure in Phase 1. A partition that
matches is evidence; a partition this module simply trusted would be decoration.

Run it:

    python3 -m codrona_mind.responses --verify
    python3 -m codrona_mind.responses --verify --cutoff 2025-01-01
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

DEFAULT_CUTOFF = "2026-01-01"

BANK_COLUMN = "in_public_problemset"
SPLIT_COLUMNS = ("user_key", "problem_key", "submitted_at", BANK_COLUMN)


class StaleArtefactError(RuntimeError):
    """The parquet on disk is not the one the sidecar manifest describes."""


def default_artefact() -> pathlib.Path:
    """The artefact path, from the same env var `lens` writes to.

    One name for one file. A second env var would let the two repos point at
    different parquets while both reported success.
    """
    from_env = os.environ.get("CODRONA_RESPONSES")
    if from_env:
        return pathlib.Path(from_env).expanduser()
    return pathlib.Path.home() / "codrona-data" / "model" / "responses.parquet"


def sidecar_manifest(artefact: pathlib.Path) -> pathlib.Path:
    """Constructed exactly as `lens` constructs it, including `with_name`."""
    return artefact.with_name(artefact.stem + ".manifest.json")


def load_manifest(artefact: pathlib.Path) -> dict[str, Any]:
    path = sidecar_manifest(artefact)
    if not path.exists():
        raise FileNotFoundError(
            f"no sidecar manifest at {path}. Regenerate it in lens: "
            "python3 -m codrona_lens.responses.matrix --real-data"
        )
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def artefact_shape(artefact: pathlib.Path) -> tuple[int, list[str]]:
    """Row count and column order from the footer. No scan."""
    parquet = pq.ParquetFile(artefact)
    return parquet.metadata.num_rows, list(parquet.schema_arrow.names)


def verify_artefact(artefact: pathlib.Path, manifest: dict[str, Any]) -> list[str]:
    """Compare the parquet on disk against the manifest that describes it."""
    if not artefact.exists():
        return [f"{artefact}: artefact absent"]
    rows, columns = artefact_shape(artefact)
    expected_rows = manifest.get("counts", {}).get("merged_responses")
    problems: list[str] = []
    if expected_rows != rows:
        problems.append(f"artefact rows: manifest {expected_rows}, file {rows}")
    block = manifest.get("schema")
    if not isinstance(block, list) or not block:
        problems.append("manifest carries no schema block")
        return problems
    expected = [str(entry["name"]) for entry in block]
    if expected != columns:
        missing = [name for name in expected if name not in columns]
        extra = [name for name in columns if name not in expected]
        for name in missing:
            problems.append(f"column {name}: in manifest, absent from artefact")
        for name in extra:
            problems.append(f"column {name}: in artefact, absent from manifest")
        if not missing and not extra:
            problems.append(f"column ORDER differs: manifest {expected}, artefact {columns}")
    return problems


def load_bank(
    artefact: pathlib.Path,
    columns: tuple[str, ...] | None = None,
    *,
    verify: bool = True,
) -> pa.Table:
    """The item bank: responses on problems in the public problemset.

    Stage A fits this and not the full matrix. `in_public_problemset` is carried
    as a column rather than baked in, so the filter happens where it is decided
    - which is here.

    Verification is ON by default because this is the read path a fit takes, and
    a check that only runs when someone remembers to invoke a command gates
    nothing. A caller wanting the rows without it passes `verify=False` and says
    so at the call site; no production path in this repo does.
    """
    if verify:
        problems = verify_artefact(artefact, load_manifest(artefact))
        if problems:
            raise StaleArtefactError(
                f"{artefact} disagrees with the manifest beside it: "
                + "; ".join(problems)
                + ". Regenerate it in lens: "
                "python3 -m codrona_lens.responses.matrix --real-data"
            )
    wanted = list(columns) if columns else None
    if wanted is not None and BANK_COLUMN not in wanted:
        wanted = [*wanted, BANK_COLUMN]
    table = pq.read_table(artefact, columns=wanted)
    return table.filter(pc.field(BANK_COLUMN))


@dataclass(frozen=True)
class Partition:
    """The evaluation split, by which gate can score each held-out response."""

    cutoff: str
    train: int
    test: int
    g1_known_user_known_item: int
    g3_new_item_only: int
    g2_new_user_only: int
    both_new: int

    @property
    def parts(self) -> int:
        return (
            self.g1_known_user_known_item
            + self.g3_new_item_only
            + self.g2_new_user_only
            + self.both_new
        )


def partition(bank: pa.Table, cutoff: str = DEFAULT_CUTOFF) -> Partition:
    """Split the bank temporally and classify every held-out response.

    A held-out response is scoreable by G1 only if both its user and its item
    appeared before the cutoff. The rest belong to G2, G3, or both.
    """
    boundary = dt.datetime.fromisoformat(cutoff)
    submitted = bank.column("submitted_at")
    scalar = pa.scalar(boundary, type=submitted.type)
    is_train = pc.less(submitted, scalar)
    train = bank.filter(is_train)
    test = bank.filter(pc.invert(is_train))
    if test.num_rows == 0:
        return Partition(cutoff, train.num_rows, 0, 0, 0, 0, 0)

    known_users = train.column("user_key").unique()
    known_items = train.column("problem_key").unique()
    user_seen = pc.is_in(test.column("user_key"), value_set=known_users)
    item_seen = pc.is_in(test.column("problem_key"), value_set=known_items)

    def count(users: bool, items: bool) -> int:
        mask = pc.and_(
            user_seen if users else pc.invert(user_seen),
            item_seen if items else pc.invert(item_seen),
        )
        return int(pc.sum(pc.cast(mask, pa.int64())).as_py() or 0)

    return Partition(
        cutoff=cutoff,
        train=train.num_rows,
        test=test.num_rows,
        g1_known_user_known_item=count(True, True),
        g3_new_item_only=count(True, False),
        g2_new_user_only=count(False, True),
        both_new=count(False, False),
    )


def verify_partition(part: Partition, manifest: dict[str, Any]) -> list[str]:
    """Two engines, two code paths, one answer.

    `lens` measures this from the warehouse in DuckDB; this measures it from the
    artefact in Arrow. Agreement is evidence about both. A disagreement means one
    of them is wrong and neither can say which, which is exactly when a build
    should stop.
    """
    recorded = manifest.get("split")
    if not isinstance(recorded, dict):
        return ["manifest carries no split block"]
    if recorded.get("cutoff") != part.cutoff:
        return [
            f"cutoff {part.cutoff} does not match the manifest's "
            f"{recorded.get('cutoff')}, so the counts describe different splits"
        ]
    problems: list[str] = []
    for name, measured in (
        ("train", part.train),
        ("test", part.test),
        ("g1_known_user_known_item", part.g1_known_user_known_item),
        ("g3_new_item_only", part.g3_new_item_only),
        ("g2_new_user_only", part.g2_new_user_only),
        ("both_new", part.both_new),
    ):
        was = recorded.get(name)
        if was != measured:
            problems.append(f"split.{name}: manifest {was}, measured here {measured}")
    if part.parts != part.test:
        problems.append(
            f"partition parts sum to {part.parts} against {part.test} held-out "
            "responses, so a response is counted twice or not at all"
        )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the response matrix this repo consumes.")
    parser.add_argument("--artefact", type=pathlib.Path, default=None)
    parser.add_argument("--cutoff", default=DEFAULT_CUTOFF)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="check shape and recompute the split against the manifest",
    )
    args = parser.parse_args(argv)

    artefact = args.artefact or default_artefact()
    if not artefact.exists():
        print(f"no artefact at {artefact} - nothing verified")
        if args.verify:
            print(
                f"--verify asked for a verification and there is nothing at "
                f"{artefact} to verify. Build it in lens: "
                "python3 -m codrona_lens.responses.matrix --real-data",
                file=sys.stderr,
            )
            return 1
        return 0
    manifest = load_manifest(artefact)

    problems = verify_artefact(artefact, manifest)
    rows, columns = artefact_shape(artefact)
    print(f"artefact  {artefact}")
    print(f"rows      {rows:,}")
    print(f"columns   {len(columns)}")

    if args.verify and not problems:
        bank = load_bank(artefact, SPLIT_COLUMNS)
        part = partition(bank, args.cutoff)
        print(f"bank      {bank.num_rows:,}")
        print(f"split     {part.cutoff}  train {part.train:,} / test {part.test:,}")
        print(f"  G1      {part.g1_known_user_known_item:,}")
        print(f"  G3      {part.g3_new_item_only:,}")
        print(f"  G2      {part.g2_new_user_only:,}")
        print(f"  both    {part.both_new:,}")
        problems += verify_partition(part, manifest)

    if problems:
        print(f"\n{len(problems)} disagreement(s):", file=sys.stderr)
        for line in problems:
            print(f"  {line}", file=sys.stderr)
        print(
            "regenerate in lens: python3 -m codrona_lens.responses.matrix --real-data",
            file=sys.stderr,
        )
        return 1
    print("\nartefact agrees with the manifest lens wrote beside it")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
