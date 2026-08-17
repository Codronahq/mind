"""Index failures produce correctly shaped arrays holding the wrong numbers.

Nothing downstream can notice that, so every guard here is tested against a
mutation that would otherwise train quietly: an unsorted axis, a key that is not
on the axis, and a prior vector offset against the items it describes.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from codrona_mind import indexing, priors

# (user, problem, rating, solved_count, contest_id, accepted)
Row = tuple[str, str, int | None, int, int | None, bool]

ROWS: list[Row] = [
    ("zoe", "300C", 2000, 500, 300, False),
    ("amy", "100A", 800, 9000, 100, True),
    ("amy", "300C", 2000, 500, 300, False),
    ("bob", "200B", 1400, 1200, 200, True),
    ("amy", "200B", 1400, 1200, 200, False),
    ("zoe", "100A", 800, 9000, 100, True),
    ("bob", "100A", 800, 9000, 100, True),
]


def _bank(rows: list[Row] | None = None) -> pa.Table:
    data = ROWS if rows is None else rows
    return pa.table(
        {
            "user_key": pa.array([r[0] for r in data], pa.string()),
            "problem_key": pa.array([r[1] for r in data], pa.string()),
            "problem_rating": pa.array([r[2] for r in data], pa.int32()),
            "solved_count": pa.array([r[3] for r in data], pa.int64()),
            "problem_contest_id": pa.array([r[4] for r in data], pa.int32()),
            "is_accepted": pa.array([r[5] for r in data], pa.bool_()),
            "in_public_problemset": pa.array([True] * len(data), pa.bool_()),
        }
    )


def _built(rows: list[Row] | None = None) -> tuple[indexing.ResponseIndex, priors.ItemCovariates]:
    bank = _bank(rows)
    items = priors.build_item_covariates(bank)
    return indexing.build_response_index(bank, items), items


def test_both_axes_are_sorted_not_encounter_ordered() -> None:
    """`unique` promises no order, and the order decides which row is whose."""
    index, _ = _built()
    assert index.user_keys == ["amy", "bob", "zoe"]
    assert index.problem_keys == ["100A", "200B", "300C"]
    assert index.n_users == 3
    assert index.n_items == 3
    assert index.n_responses == len(ROWS)


def test_the_index_is_stable_under_input_order() -> None:
    """Two runs on the same data must produce comparable parameter vectors."""
    forward, _ = _built()
    backward, _ = _built(list(reversed(ROWS)))
    assert forward.user_keys == backward.user_keys
    assert forward.problem_keys == backward.problem_keys
    forward_pairs = zip(forward.user_index.tolist(), forward.item_index.tolist(), strict=True)
    backward_pairs = zip(backward.user_index.tolist(), backward.item_index.tolist(), strict=True)
    assert sorted(forward_pairs) == sorted(backward_pairs)


def test_positions_point_at_the_right_keys() -> None:
    """The whole module is wrong if this mapping is off, and nothing else checks it."""
    index, _ = _built()
    for row, user_pos, item_pos, correct in zip(
        ROWS, index.user_index, index.item_index, index.correct, strict=True
    ):
        assert index.user_keys[user_pos] == row[0]
        assert index.problem_keys[item_pos] == row[1]
        assert bool(correct) is row[5]


def test_counts_per_axis_are_right() -> None:
    index, _ = _built()
    per_user = dict(zip(index.user_keys, index.responses_per_user().tolist(), strict=True))
    per_item = dict(zip(index.problem_keys, index.responses_per_item().tolist(), strict=True))
    assert per_user == {"amy": 3, "bob": 2, "zoe": 2}
    assert per_item == {"100A": 3, "200B": 2, "300C": 2}
    assert per_user["amy"] + per_user["bob"] + per_user["zoe"] == index.n_responses


def test_the_item_axis_comes_from_the_covariates_not_the_responses() -> None:
    """An item with no responses keeps its parameter slot.

    Deriving the axis from responses would make the vector change length with
    whichever subset is being fitted, so a train-split fit and a full fit would
    not be comparable.
    """
    bank = _bank()
    items = priors.build_item_covariates(bank)
    subset = bank.filter(pa.array([r[1] != "300C" for r in ROWS]))
    index = indexing.build_response_index(subset, items)
    assert index.problem_keys == ["100A", "200B", "300C"]
    assert index.responses_per_item().tolist() == [3, 2, 0]
    indexing.align_items(index, items)


def test_a_key_absent_from_the_axis_raises() -> None:
    """`index_in` emits null, and a null cast to an integer is position zero.

    Position zero is a real user and a real item, so the fit would train the
    wrong parameters rather than fail.
    """
    bank = _bank()
    items = priors.build_item_covariates(_bank([r for r in ROWS if r[1] != "300C"]))
    with pytest.raises(KeyError, match="300C"):
        indexing.build_response_index(bank, items)


def test_an_unsorted_covariate_frame_raises() -> None:
    """A prior offset against its items trains quietly and predicts nonsense."""
    index, items = _built()
    shuffled = items.table.take(pa.array([2, 0, 1]))
    broken = priors.ItemCovariates(
        table=shuffled,
        rating=items.rating,
        log_solved=items.log_solved,
        imputed_rating=items.imputed_rating,
        unrated_items=items.unrated_items,
        zero_solved_items=items.zero_solved_items,
        items_without_contest_id=items.items_without_contest_id,
    )
    with pytest.raises(ValueError, match="sorted key order"):
        indexing.build_response_index(_bank(), broken)
    with pytest.raises(ValueError, match="item axis disagrees"):
        indexing.align_items(index, broken)


def test_align_items_accepts_a_matching_frame() -> None:
    index, items = _built()
    indexing.align_items(index, items)  # must not raise


def test_a_bank_missing_fit_columns_names_all_of_them() -> None:
    """Two missing at once is what distinguishes this check from pyarrow's.

    Dropping one column raises KeyError from `bank.column` anyway, naming it -
    so a single-column test passes with the check deleted and gates nothing.
    The check exists to report every missing column before any work starts, and
    that is what this asserts.
    """
    bank = _bank()
    items = priors.build_item_covariates(bank)
    broken = bank.drop_columns(["is_accepted", "user_key"])
    with pytest.raises(KeyError) as caught:
        indexing.build_response_index(broken, items)
    message = str(caught.value)
    assert "is_accepted" in message
    assert "user_key" in message


def test_describe_reports_the_shape_the_fit_will_see() -> None:
    lines = "\n".join(_built()[0].describe())
    assert "responses          7" in lines
    assert "users              3" in lines
    assert "items              3" in lines
    assert "correct            4" in lines


def test_index_dtypes_are_what_a_fit_can_gather_with() -> None:
    """Float indices or object arrays would fail deep inside the fit, if at all."""
    index, _ = _built()
    assert index.user_index.dtype == np.int64
    assert index.item_index.dtype == np.int64
    assert index.correct.dtype == np.bool_
    assert index.user_index.shape == index.item_index.shape == index.correct.shape
