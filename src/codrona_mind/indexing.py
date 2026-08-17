"""Map the response matrix onto contiguous integer indices for the fit.

A 2PL fit holds one ability per user and two parameters per item, addressed by
position in an array. Getting from `user_key` and `problem_key` to those
positions is the least interesting code in Stage A and among the easiest to get
silently wrong, because every failure here produces arrays of the right shape
holding the wrong numbers.

DETERMINISM IS THE WHOLE POINT. The order of these indices decides which row of
the parameter array belongs to which person, so an order that varies between
runs makes two fits on identical data incomparable - and nothing about the
output looks wrong. Arrow's `unique` and `group_by` promise no order at all, so
both axes are sorted explicitly. `priors.build_item_covariates` sorts its items
the same way and for the same reason, and `align_items` below asserts the two
orderings agree rather than assuming it: a prior vector offset by one row
against the items it describes would train quietly and predict nonsense.

THE ITEM AXIS COMES FROM THE COVARIATES, NOT FROM THE RESPONSES. Every bank item
has covariates, but an item can carry responses in one split and none in
another, so deriving the axis from whichever responses are in hand would make
the parameter vector change length with the subset being fitted. The covariate
frame is the authority; responses index into it.

WHAT THIS DOES NOT DO. It does not filter, subset, or split. The bank filter
lives in `responses.load_bank` and the temporal split in `responses.partition`,
both of which are already gated against `lens`. Adding a second place that
decides which responses are fitted is how the two start disagreeing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc

from codrona_mind.priors import PROBLEM_COLUMN, ItemCovariates

USER_COLUMN = "user_key"
ACCEPTED_COLUMN = "is_accepted"

FIT_COLUMNS = (USER_COLUMN, PROBLEM_COLUMN, ACCEPTED_COLUMN)

IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class ResponseIndex:
    """Responses as integer positions, plus the keys those positions mean."""

    user_keys: list[str]
    problem_keys: list[str]
    user_index: IntArray
    item_index: IntArray
    correct: BoolArray

    @property
    def n_users(self) -> int:
        return len(self.user_keys)

    @property
    def n_items(self) -> int:
        return len(self.problem_keys)

    @property
    def n_responses(self) -> int:
        return int(self.user_index.shape[0])

    def responses_per_user(self) -> IntArray:
        return np.bincount(self.user_index, minlength=self.n_users).astype(np.int64)

    def responses_per_item(self) -> IntArray:
        return np.bincount(self.item_index, minlength=self.n_items).astype(np.int64)

    def describe(self) -> list[str]:
        per_item = self.responses_per_item()
        per_user = self.responses_per_user()
        return [
            f"responses          {self.n_responses:,}",
            f"users              {self.n_users:,}",
            f"items              {self.n_items:,}",
            f"correct            {int(self.correct.sum()):,}"
            f" ({100.0 * float(self.correct.mean()):.4f}%)",
            f"items with none    {int((per_item == 0).sum()):,}",
            f"users with none    {int((per_user == 0).sum()):,}",
            f"median per item    {int(np.median(per_item))}",
            f"median per user    {int(np.median(per_user))}",
        ]


def _sorted_unique(values: pa.ChunkedArray | pa.Array) -> list[str]:
    """Explicitly sorted, because `unique` guarantees nothing about order."""
    unique = pc.unique(values)
    ordered = unique.take(pc.sort_indices(unique))
    return [str(value) for value in ordered.to_pylist()]


def _positions(values: pa.ChunkedArray | pa.Array, keys: list[str]) -> IntArray:
    """Position of every value in `keys`, raising on anything not present.

    `index_in` emits null for an unmatched value, and a null silently cast to an
    integer becomes a position - usually zero, which is a real user and a real
    item. That would train the wrong parameters rather than fail.
    """
    table = pa.array(keys, type=pa.string())
    matched = pc.index_in(values, value_set=table)
    if pc.sum(pc.cast(pc.is_null(matched), pa.int64())).as_py():
        missing = pc.unique(pc.filter(values, pc.is_null(matched))).to_pylist()
        raise KeyError(f"{len(missing)} key(s) absent from the axis: {missing[:5]}")
    return np.asarray(matched.to_numpy(zero_copy_only=False), dtype=np.int64)


def build_response_index(bank: pa.Table, items: ItemCovariates) -> ResponseIndex:
    """Index a response table against an item axis the covariates define."""
    missing = [name for name in FIT_COLUMNS if name not in bank.schema.names]
    if missing:
        raise KeyError(f"bank is missing fit columns: {missing}")

    problem_keys = [str(key) for key in items.table.column(PROBLEM_COLUMN).to_pylist()]
    if problem_keys != sorted(problem_keys):
        raise ValueError(
            "item covariates are not in sorted key order, so the parameter "
            "vector would not line up with the prior describing it"
        )

    user_keys = _sorted_unique(bank.column(USER_COLUMN))
    user_index = _positions(bank.column(USER_COLUMN), user_keys)
    item_index = _positions(bank.column(PROBLEM_COLUMN), problem_keys)
    correct = np.asarray(
        bank.column(ACCEPTED_COLUMN).to_numpy(zero_copy_only=False), dtype=np.bool_
    )
    return ResponseIndex(
        user_keys=user_keys,
        problem_keys=problem_keys,
        user_index=user_index,
        item_index=item_index,
        correct=correct,
    )


def align_items(index: ResponseIndex, items: ItemCovariates) -> None:
    """Assert the parameter axis and the prior describe the same items, in order.

    Cheap, and the failure it guards is not: a prior vector offset against the
    items it describes trains without complaint and predicts nonsense, with no
    shape mismatch anywhere to notice.
    """
    covariate_keys = [str(key) for key in items.table.column(PROBLEM_COLUMN).to_pylist()]
    if covariate_keys != index.problem_keys:
        raise ValueError(
            f"item axis disagrees with the covariate frame: "
            f"{len(index.problem_keys)} indexed against {len(covariate_keys)} "
            "described, or the same count in a different order"
        )
