"""The in-place compaction commit archives exactly the rows the compressor was handed (#121734).

#121302 capped the archive at the newest row the compacting surface held, which still archives by position:
a foreign-row gap below that row, or the lease watermark reached through an unpersisted current turn, took
turns the summary never contained. The commit now names the held rows by id from the pre-dispatch snapshot
(``messages_before_compression``), archives those, clones every other active row after the summary, and
refuses when a held row is no longer active. The reproductions are @ehz0ah's from the #120156 review.
"""

from __future__ import annotations

import copy
from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import _DB_PERSISTED_MARKER
from tests.agent.test_conversation_compression_manual import (
    FOREIGN_TURN, _compress, _exchanges, _flags, _stored_agent, session_db,  # noqa: F401  (fixture)
)

SUMMARY_TEXT = "Numbered fruit questions"


def _live_contents(db):
    return [m["content"] for m in db.get_messages_as_conversation("sid")]


def _archived(db, content):
    """The row(s) with this content were summarized away (``active=0, compacted=1``)."""
    return (0, 1) in _flags(db, content)


def _held_with_own_turn(db, held, turn):
    """A surface that persisted its own turn after loading: the dicts carry the row id and persist marker,
    as ``sync_flushed_message_markers`` stamps them after a flush."""
    for role, content in turn:
        rid = db.append_message("sid", role, content)
        held.append({"role": role, "content": content, "_row_id": rid, _DB_PERSISTED_MARKER: True})


OWN_TURN = [("user", "question 99 about my own later turn"), ("assistant", "answer 99 to the later turn")]


def test_foreign_rows_in_a_gap_below_the_newest_held_row_stay_live(session_db):
    """Case 1: the surface holds rows 1-20, misses another surface's 21-22, then persists its own 23-24.
    Capping at max(held)=24 archived 21-22 unseen; archiving by held id leaves them live after the summary."""
    agent, _ = _stored_agent(session_db, _exchanges(10))
    held = session_db.get_resume_conversations("sid")[0]
    for role, content in FOREIGN_TURN:
        session_db.append_message("sid", role, content)
    _held_with_own_turn(session_db, held, OWN_TURN)
    assert [m["_row_id"] for m in held[-2:]] == [23, 24]

    assert _compress(agent, held, "").status == "compressed"

    live = _live_contents(session_db)
    for _role, content in FOREIGN_TURN:
        assert live.count(content) == 1
        assert not _archived(session_db, content)
    assert len(live) == len(set(live))
    assert any(SUMMARY_TEXT in c for c in live)
    assert session_db.search_messages("vault 7741")


def test_foreign_rows_appended_since_load_stay_live_under_an_unpersisted_current_turn(session_db):
    """Case 2: turn preflight appends the current user message before compression and persists it after, so
    the trailing held dict has no row id. That must not widen the archive to the lease watermark: rows another
    surface appended since load were never summarized and stay live."""
    agent, _ = _stored_agent(session_db, _exchanges(10))
    held = session_db.get_resume_conversations("sid")[0]
    for role, content in FOREIGN_TURN:
        session_db.append_message("sid", role, content)
    held.append({"role": "user", "content": "current turn, not yet persisted 5150"})
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = f"## Goal\n{SUMMARY_TEXT}.\n## Progress\nEarly ones answered."

    with patch("agent.context_compressor.call_llm", lambda **_kw: response):
        compressed, _ = agent._compress_context(held, "", force=True)

    assert len(compressed) < len(held)
    live = _live_contents(session_db)
    for _role, content in FOREIGN_TURN:
        assert live.count(content) == 1
        assert not _archived(session_db, content)
    assert live.count("current turn, not yet persisted 5150") == 1
    assert len(live) == len(set(live))


def test_in_place_engine_boundary_comes_from_the_pre_dispatch_snapshot(session_db):
    """Case 3: a valid ``ContextEngine`` may mutate the list it is handed: reuse the first loaded dict (marker
    and row id intact) as its summary and truncate the rest. The held boundary must be read before dispatch,
    so rows 2-20 are archived and no live copy stands beside the summary."""
    agent, _ = _stored_agent(session_db, _exchanges(10))
    held = session_db.get_resume_conversations("sid")[0]
    before = copy.deepcopy(held)

    def engine_compress(messages, **_kwargs):
        summary = messages[0]
        summary["content"] = f"## Goal\n{SUMMARY_TEXT}.\n## Progress\nrows 1-20 summarized in place."
        del messages[1:]
        return messages

    agent.context_compressor.compress = engine_compress
    assert _compress(agent, held, "").status == "compressed"

    live = _live_contents(session_db)
    assert len(live) == 1 and SUMMARY_TEXT in live[0]
    for message in before[1:]:
        assert _flags(session_db, message["content"]) == [(0, 1)]
    assert _flags(session_db, before[0]["content"]) in ([(0, 0)], [(0, 1)])


def test_merged_rows_are_archived_and_foreign_rows_stay_live(session_db):
    """Case 4: resume repair merged two adjacent user rows into one dict, which keeps only the first row's id
    and drops the persist marker. The merged dict must name both source rows so both are archived, while rows
    another surface appended after the load stay live."""
    agent, _ = _stored_agent(session_db, _exchanges(10))
    session_db.append_message("sid", "user", "first half of a split prompt")
    session_db.append_message("sid", "user", "second half 9931")
    held = session_db.get_resume_conversations("sid")[0]  # the resume path merges the two user rows
    assert "second half 9931" in held[-1]["content"] and held[-1]["content"] != "second half 9931"
    for role, content in FOREIGN_TURN:
        session_db.append_message("sid", role, content)

    assert _compress(agent, held, "").status == "compressed"

    live = _live_contents(session_db)
    assert sum("second half 9931" in c for c in live) == 1
    assert (1, 0) not in _flags(session_db, "second half 9931")
    assert (1, 0) not in _flags(session_db, "first half of a split prompt")
    for _role, content in FOREIGN_TURN:
        assert live.count(content) == 1
        assert not _archived(session_db, content)


def test_commit_is_refused_when_a_held_row_is_no_longer_active(session_db):
    """Case 5: another surface compacted the session between this surface's load and its commit, so the held
    rows are archived. Committing anyway would archive the winner's rows and clone them back beside a second
    summary. The commit is refused: nothing is archived, no summary row is inserted, the live list is
    unchanged, and the surface is told."""
    agent, _ = _stored_agent(session_db, _exchanges(10))
    held = session_db.get_resume_conversations("sid")[0]
    other, _ = _stored_agent(session_db, [], create=False)
    assert _compress(other, session_db.get_messages_as_conversation("sid"), "").status == "compressed"
    durable_before = session_db.get_messages_as_conversation("sid")
    frozen = copy.deepcopy(held)
    agent._emit_warning = MagicMock()

    result = _compress(agent, held, "")

    assert result.after_messages == frozen and held == frozen
    assert session_db.get_messages_as_conversation("sid") == durable_before
    assert sum(SUMMARY_TEXT in c for c in _live_contents(session_db)) == 1
    all_rows = session_db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = 'sid' AND content LIKE ?", (f"%{SUMMARY_TEXT}%",)).fetchone()[0]
    assert all_rows == 1, "the refused commit must not insert a second summary"
    assert agent._emit_warning.called
    assert "refused" in " ".join(str(c.args[0]) for c in agent._emit_warning.call_args_list).lower()


@pytest.mark.parametrize("raw", ["", "here 2"])
def test_gateway_replay_without_row_ids_keeps_the_watermark_commit(session_db, raw):
    """The messaging gateway replays history without row ids (``load_transcript``). With no held id to
    archive by, the commit keeps the watermark rule from #121302 and still stores one live copy of each row."""
    agent, _ = _stored_agent(session_db, _exchanges(10))
    held = session_db.get_messages_as_conversation("sid", repair_alternation=True)
    assert all("_row_id" not in m for m in held)

    assert _compress(agent, held, raw).status == "compressed"

    live = _live_contents(session_db)
    assert len(live) == len(set(live))
    assert any(SUMMARY_TEXT in c for c in live)
