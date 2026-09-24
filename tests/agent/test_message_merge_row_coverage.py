"""Merge coverage: a dict that absorbed another message names every durable row it now stands for (#121734).

``repair_message_sequence`` merges adjacent same-role rows into the first dict, which keeps only its own
``_row_id``; the absorbed row's id left the live history, so an in-place compaction that summarized the merged
dict could not archive the absorbed row and cloned it back beside a summary that already contained it. The
merge passes (and the ``here N`` seam fold) now record ``_row_ids``; ``message_row_ids`` reads both keys.
"""

from __future__ import annotations

from agent.agent_runtime_helpers import _merge_consecutive_assistants, _merge_consecutive_users
from agent.context_compressor import _DB_PERSISTED_MARKER
from agent.message_metadata import PERSISTENCE_ONLY_MESSAGE_FIELDS, message_row_ids, record_merge_coverage
from hermes_cli.partial_compress import rejoin_compressed_head_and_tail


def _row(role, content, rid):
    return {"role": role, "content": content, "_row_id": rid, _DB_PERSISTED_MARKER: True}


def test_merged_user_rows_name_both_source_rows():
    merged, repairs = _merge_consecutive_users([_row("user", "first half", 21), _row("user", "second half", 22)])
    assert repairs == 1 and len(merged) == 1
    assert merged[0]["_row_id"] == 21 and _DB_PERSISTED_MARKER not in merged[0]
    assert message_row_ids(merged[0]) == [21, 22]


def test_merged_assistant_rows_name_both_source_rows():
    merged, repairs = _merge_consecutive_assistants([_row("assistant", "part one", 7), _row("assistant", "part two", 8)])
    assert repairs == 1 and len(merged) == 1
    assert message_row_ids(merged[0]) == [7, 8]


def test_a_superseded_verification_candidate_is_still_covered():
    provisional = {**_row("assistant", "provisional", 7), "finish_reason": "verification_required"}
    merged, _ = _merge_consecutive_assistants([provisional, _row("assistant", "final", 8)])
    assert merged[0]["content"] == "final" and message_row_ids(merged[0]) == [8, 7]


def test_a_third_merge_accumulates_coverage_and_an_unpersisted_row_adds_nothing():
    rows = [_row("user", "a", 1), _row("user", "b", 2), {"role": "user", "content": "c"}]
    merged, _ = _merge_consecutive_users(rows)
    assert message_row_ids(merged[0]) == [1, 2]
    assert message_row_ids({"role": "user", "content": "never stored"}) == []
    assert message_row_ids({"role": "user", "content": "x", "_row_id": True, "_row_ids": [0, -1, "3", 4.0, 5]}) == [5]


def test_dicts_that_never_touched_the_db_stay_free_of_the_key():
    a, b = {"role": "user", "content": "a"}, {"role": "user", "content": "b"}
    record_merge_coverage(a, b)
    assert "_row_ids" not in a


def test_seam_fold_names_the_folded_tail_row():
    head = [{"role": "user", "content": "summary", "_row_id": 3}]
    tail = [_row("user", "kept question", 9), _row("assistant", "kept answer", 10)]
    joined = rejoin_compressed_head_and_tail(head, tail)
    assert len(joined) == 2 and joined[0]["content"] == "summary\n\nkept question"
    assert message_row_ids(joined[0]) == [3, 9]
    assert "_row_ids" not in head[0], "the fold builds a new dict; the head's own is untouched"


def test_merge_coverage_never_reaches_the_provider():
    assert "_row_ids" in PERSISTENCE_ONLY_MESSAGE_FIELDS and "_row_id" in PERSISTENCE_ONLY_MESSAGE_FIELDS
