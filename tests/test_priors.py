"""Every edge case this module names must be exercised, not just handled.

The fixture deliberately contains a zero `solved_count`, an unrated problem, a
problem with no contest id, and a problem appearing on several responses - so no
branch passes by never being reached, which is how the observatory's NULL
columns went unnoticed through 159 passing tests.
"""

from __future__ import annotations

import math
import random
import statistics

import pyarrow as pa
import pytest

from codrona_mind import priors

# (problem, rating, solved_count, contest_id)
Item = tuple[str, int | None, int, int | None]

ITEMS: list[Item] = [
    ("100A", 800, 5000, 100),
    ("200B", 1600, 900, 200),
    ("300C", 2900, 0, 300),  # zero solved_count: log(0) would be -inf
    ("400D", None, 40, 400),  # unrated: rating must be imputed, flagged
    ("SGU1", None, 12, None),  # acmsguru: no rating and no contest id
]


def _bank(items: list[Item] | None = None, repeats: int = 3) -> pa.Table:
    """One item can appear on many responses; the collapse must be idempotent."""
    data = ITEMS if items is None else items
    rows = [item for item in data for _ in range(repeats)]
    return pa.table(
        {
            "problem_key": pa.array([r[0] for r in rows], pa.string()),
            "problem_rating": pa.array([r[1] for r in rows], pa.int32()),
            "solved_count": pa.array([r[2] for r in rows], pa.int64()),
            "problem_contest_id": pa.array([r[3] for r in rows], pa.int32()),
            "in_public_problemset": pa.array([True] * len(rows), pa.bool_()),
        }
    )


def test_responses_collapse_to_one_row_per_problem() -> None:
    cov = priors.build_item_covariates(_bank(repeats=7))
    assert cov.n_items == len(ITEMS)
    assert cov.table.column("problem_key").to_pylist() == sorted(item[0] for item in ITEMS)


def test_the_item_order_is_deterministic() -> None:
    """An unordered index makes two runs produce different parameter vectors."""
    forward = priors.build_item_covariates(_bank())
    backward = priors.build_item_covariates(_bank(list(reversed(ITEMS))))
    assert (
        forward.table.column("problem_key").to_pylist()
        == backward.table.column("problem_key").to_pylist()
    )


def test_a_zero_solved_count_stays_finite() -> None:
    """The 31 real bank items at zero. log(0) is -inf; log1p(0) is 0."""
    cov = priors.build_item_covariates(_bank())
    values = cov.table.column("log_solved_z").to_pylist()
    assert cov.zero_solved_items == 1
    assert all(value is not None and math.isfinite(value) for value in values)


def test_an_unrated_item_is_imputed_and_flagged() -> None:
    """Imputing without a flag would tell the prior these are averagely hard."""
    cov = priors.build_item_covariates(_bank())
    keys = cov.table.column("problem_key").to_pylist()
    flags = cov.table.column("is_unrated").to_pylist()
    flagged = {key for key, flag in zip(keys, flags, strict=True) if flag}
    assert flagged == {"400D", "SGU1"}
    assert cov.unrated_items == 2
    ratings = cov.table.column("rating_z").to_pylist()
    assert all(value is not None and math.isfinite(value) for value in ratings)


def test_the_imputed_rating_is_the_median_of_rated_items() -> None:
    """Rating is bounded at 800 and long-tailed, so a mean sits above typical."""
    cov = priors.build_item_covariates(_bank())
    assert cov.imputed_rating == 1600.0


def _grid_bank(n: int, seed: int, chunks: int = 1) -> pa.Table:
    """Ratings on the 100-point grid Codeforces uses, optionally chunked."""
    rng = random.Random(seed)
    keys = [f"P{i:05d}" for i in range(n)]
    ratings = [rng.randrange(800, 3501, 100) for _ in range(n)]
    table = pa.table(
        {
            "problem_key": pa.array(keys, pa.string()),
            "problem_rating": pa.array(ratings, pa.int32()),
            "solved_count": pa.array([100] * n, pa.int64()),
            "problem_contest_id": pa.array(list(range(n)), pa.int32()),
            "in_public_problemset": pa.array([True] * n, pa.bool_()),
        }
    )
    if chunks > 1:
        step = max(1, n // chunks)
        batches = [table.slice(i, step) for i in range(0, n, step)]
        table = pa.concat_tables(batches)
    return table


@pytest.mark.parametrize("seed", [0, 2, 3, 4])
def test_the_imputed_rating_is_exact_at_scale(seed: int) -> None:
    """A t-digest misses the median, and five fixture items cannot see it.

    SEEDS CHOSEN BY MEASUREMENT, NOT BY HOPE. At n=400 the t-digest deviates on
    11 of 12 seeds - and the first version of this test used seed 1, the one
    that happens to land exactly, so the mutation reverting to
    approximate_median survived. Each seed here was checked to deviate.
    """
    bank = _grid_bank(400, seed=seed)
    ratings = bank.column("problem_rating").to_pylist()
    cov = priors.build_item_covariates(bank)
    assert cov.imputed_rating == statistics.median(ratings)


def test_the_imputed_rating_does_not_depend_on_chunking() -> None:
    """INERT AGAINST THE T-DIGEST, and kept anyway with that said out loud.

    `approximate_median` is chunk-unstable in general - 1816.92 as one chunk
    against 1798.32 as three - but the `group_by` inside the builder emits a
    single chunk regardless of input layout, so that instability cannot reach
    the imputed value and this test cannot fail because of it. Measured.

    It stays because chunk-independence of the emitted covariates is a real
    property worth holding as the builder grows, and a reader should know
    exactly which protection it does and does not provide.
    """
    one = priors.build_item_covariates(_grid_bank(400, seed=2, chunks=1))
    many = priors.build_item_covariates(_grid_bank(400, seed=2, chunks=5))
    assert one.imputed_rating == many.imputed_rating
    assert one.rating.mean == many.rating.mean
    assert one.rating.sd == many.rating.sd


def test_a_missing_contest_id_is_counted_not_dropped() -> None:
    """acmsguru items simply fall out of the post-fit date diagnostic."""
    cov = priors.build_item_covariates(_bank())
    assert cov.items_without_contest_id == 1
    contest = dict(
        zip(
            cov.table.column("problem_key").to_pylist(),
            cov.table.column("contest_id").to_pylist(),
            strict=True,
        )
    )
    assert contest["SGU1"] is None
    assert contest["100A"] == 100


def test_covariates_are_standardised_and_the_centring_is_recorded() -> None:
    """A fit that standardised differently at inference would be wrong quietly."""
    cov = priors.build_item_covariates(_bank())
    for column, scale in (
        ("rating_z", cov.rating),
        ("log_solved_z", cov.log_solved),
    ):
        values = [v for v in cov.table.column(column).to_pylist() if v is not None]
        assert abs(sum(values) / len(values)) < 1e-9
        assert scale.sd > 0
    # The recorded centring must invert back to the raw values it came from.
    raw = [800.0, 1600.0, 2900.0, 1600.0, 1600.0]
    recovered = [
        value * cov.rating.sd + cov.rating.mean
        for value in cov.table.column("rating_z").to_pylist()
    ]
    assert all(abs(a - b) < 1e-9 for a, b in zip(sorted(recovered), sorted(raw), strict=True))


def test_a_constant_covariate_does_not_divide_by_zero() -> None:
    """One item, or identical ratings: sd is zero and inf would reach the prior."""
    flat: list[Item] = [("A", 1500, 10, 1), ("B", 1500, 10, 2)]
    cov = priors.build_item_covariates(_bank(flat))
    values = cov.table.column("rating_z").to_pylist()
    assert all(value == 0.0 for value in values)
    assert cov.rating.sd == 1.0


def test_a_bank_missing_an_item_column_raises() -> None:
    """Better than emitting a prior built from columns that were not there."""
    bank = _bank().drop_columns(["solved_count"])
    with pytest.raises(KeyError, match="solved_count"):
        priors.build_item_covariates(bank)


def test_describe_reports_every_edge_case_count() -> None:
    """These counts are the reason to read the output at all."""
    lines = "\n".join(priors.build_item_covariates(_bank()).describe())
    assert "items                      5" in lines
    assert "unrated, rating imputed    2" in lines
    assert "solved_count zero          1" in lines
    assert "no contest id, no date     1" in lines
