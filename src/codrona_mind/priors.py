"""Per-item covariates for the difficulty prior, with every edge case named.

Stage A does not fit a free 2PL. `architecture/phase-2-modelling.md` draws item
difficulty from a prior conditioned on `problem_rating` and `log(solved_count)`,
so a thin item shrinks toward its prior instead of fitting noise. This module
builds that conditioning, one row per problem, and it is deliberately separate
from the fit: the coefficients relating these covariates to difficulty are
LEARNED, not set here. What is set here is what the covariates are, what happens
when they are missing, and what is carried alongside so the fit can be audited.

THREE EDGE CASES, ALL MEASURED RATHER THAN ASSUMED.

`solved_count` is zero on 31 of the 11,764 bank problems - 15 rated 2800+ and 16
unrated. `log(0)` is negative infinity, and a prior mean of negative infinity
silently poisons whatever it touches. The blueprint specified no guard. This
module uses `log1p`, which is exact at zero rather than approximately handled,
and carries no offset a reader has to know about.

`problem_rating` is null on 713 bank problems. Dropping them would remove items
Stage A is meant to fit; imputing silently would tell the prior those problems
are averagely hard when nothing is known about them. Both the imputed value and
an indicator column are emitted, so the fit can learn a separate offset for
"unrated" rather than inheriting a fiction.

`problem_contest_id` is null on 417 bank problems, all unrated, all from the
acmsguru archive. It is not a prior covariate - it is carried for the post-fit
diagnostic the blueprint requires, correlating fitted difficulty residual
against publication date, which is the only thing that separates a contaminated
covariate from genuine drift in Codeforces' own calibration. An item with no
contest id simply drops out of that diagnostic, and the count is reported rather
than left to be discovered.

STANDARDISATION IS RECORDED, NOT RECOMPUTED. The mean and standard deviation
used to centre each covariate are returned with the frame. A fit that
standardised at training time and not at inference time, or standardised against
a different population, would be wrong in a way no shape check could catch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.compute as pc

RATING_COLUMN = "problem_rating"
SOLVED_COLUMN = "solved_count"
CONTEST_COLUMN = "problem_contest_id"
PROBLEM_COLUMN = "problem_key"

ITEM_COLUMNS = (
    PROBLEM_COLUMN,
    RATING_COLUMN,
    SOLVED_COLUMN,
    CONTEST_COLUMN,
    "in_public_problemset",
)


@dataclass(frozen=True)
class Standardisation:
    """The centring used, carried so inference cannot silently differ."""

    mean: float
    sd: float

    def apply(self, values: pa.Array) -> pa.Array:
        return pc.divide(pc.subtract(values, self.mean), self.sd)


@dataclass(frozen=True)
class ItemCovariates:
    """One row per problem, in a fixed order, plus what was done to build it."""

    table: pa.Table
    rating: Standardisation
    log_solved: Standardisation
    imputed_rating: float
    unrated_items: int
    zero_solved_items: int
    items_without_contest_id: int

    @property
    def n_items(self) -> int:
        return int(self.table.num_rows)

    def describe(self) -> list[str]:
        return [
            f"items                      {self.n_items:,}",
            f"unrated, rating imputed    {self.unrated_items:,}"
            f" (imputed {self.imputed_rating:.1f})",
            f"solved_count zero          {self.zero_solved_items:,}",
            f"no contest id, no date     {self.items_without_contest_id:,}"
            " (excluded from the post-fit date diagnostic)",
            f"rating centring            mean {self.rating.mean:.4f} sd {self.rating.sd:.4f}",
            f"log1p(solved) centring     mean {self.log_solved.mean:.4f}"
            f" sd {self.log_solved.sd:.4f}",
        ]


def _standardise(values: pa.Array) -> tuple[pa.Array, Standardisation]:
    mean = float(pc.mean(values).as_py())
    sd = float(pc.stddev(values, ddof=1).as_py() or 0.0)
    if sd == 0.0 or math.isnan(sd):
        # A constant covariate carries no information and dividing by zero
        # would emit inf or nan into the prior. Centre and leave the scale
        # alone; the fit sees a column of zeros, which is honest.
        sd = 1.0
    scale = Standardisation(mean=mean, sd=sd)
    return scale.apply(values), scale


def build_item_covariates(bank: pa.Table) -> ItemCovariates:
    """Collapse a response table to one row per problem, with covariates ready.

    Takes the bank rather than the full matrix: Stage A fits the item bank, and
    an item outside it has no place in a prior for items inside it.
    """
    columns = [name for name in ITEM_COLUMNS if name in bank.schema.names]
    missing = [name for name in ITEM_COLUMNS[:4] if name not in bank.schema.names]
    if missing:
        raise KeyError(f"bank is missing item columns: {missing}")

    items = (
        bank.select(columns)
        .group_by(PROBLEM_COLUMN)
        .aggregate(
            [
                (RATING_COLUMN, "max"),
                (SOLVED_COLUMN, "max"),
                (CONTEST_COLUMN, "max"),
            ]
        )
    )
    # group_by does not promise an order, and an unordered item index would make
    # two runs produce different parameter vectors for the same data.
    items = items.sort_by([(PROBLEM_COLUMN, "ascending")])

    rating = items.column(f"{RATING_COLUMN}_max")
    solved = items.column(f"{SOLVED_COLUMN}_max")
    contest = items.column(f"{CONTEST_COLUMN}_max")

    unrated = int(pc.sum(pc.cast(pc.is_null(rating), pa.int64())).as_py() or 0)
    zero_solved = int(pc.sum(pc.cast(pc.equal(solved, 0), pa.int64())).as_py() or 0)
    no_contest = int(pc.sum(pc.cast(pc.is_null(contest), pa.int64())).as_py() or 0)

    # The median of the rated items, not the mean: rating is bounded below at
    # 800 and long-tailed above, so a mean sits higher than any typical item.
    #
    # `quantile` and not `approximate_median`. The latter is a t-digest and is
    # simply inexact: over 400 grid-valued ratings it misses the true median on
    # 11 of 12 seeds, by up to 40 points, and over the real 11,051 it returned
    # 1828.5 - not on the 100-point grid Codeforces ratings occupy, which is the
    # only reason it was noticed at all. The imputed value feeds the centring,
    # so every standardised rating in the bank inherits the error.
    #
    # It is also chunk-unstable in general - 1816.92 as one chunk against
    # 1798.32 as three, for identical values - but that hazard does NOT reach
    # here, because the group_by above emits a single chunk regardless of how
    # the input was laid out. Measured, not assumed, and stated so nobody
    # defends this line for a reason that does not apply.
    imputed_quantile = pc.quantile(pc.drop_null(rating), q=0.5)
    imputed = float(imputed_quantile[0].as_py() or 0.0)
    rating_filled = pc.fill_null(pc.cast(rating, pa.float64()), imputed)

    # log1p rather than log(x + eps): exact at zero, no offset a reader has to
    # know about, and the 31 zero-count items keep a finite prior mean.
    log_solved = pc.log1p(pc.cast(solved, pa.float64()))

    rating_z, rating_scale = _standardise(rating_filled)
    solved_z, solved_scale = _standardise(log_solved)

    table = pa.table(
        {
            PROBLEM_COLUMN: items.column(PROBLEM_COLUMN),
            "rating_z": rating_z,
            "log_solved_z": solved_z,
            "is_unrated": pc.is_null(rating),
            "contest_id": contest,
        }
    )
    return ItemCovariates(
        table=table,
        rating=rating_scale,
        log_solved=solved_scale,
        imputed_rating=imputed,
        unrated_items=unrated,
        zero_solved_items=zero_solved,
        items_without_contest_id=no_contest,
    )
