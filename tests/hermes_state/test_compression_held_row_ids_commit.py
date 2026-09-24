"""``archive_and_compact(held_row_ids=...)``: the in-place commit archives exactly the rows it was handed (#121734).

The watermark commit (#75316, #121302) archives by position: every active row at or below a cap. A surface that
holds rows another surface interleaved with (a gap below its newest row) or that reached the lease watermark
through an unpersisted current turn archived rows the summary never saw. With ``held_row_ids`` the commit is
exact: those rows are archived, the ones a compacted dict still names are rewound, every other active row is
cloned after the compacted set, and a held row that is no longer active refuses the whole commit.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_state import SessionDB
from hermes_state_errors import StaleHeldRowsError


@pytest.fixture
def db(tmp_path: Path) -> SessionDB:
    d = SessionDB(tmp_path / "state.db")
    d.create_session("sess1", source="test")
    yield d
    d.close()


def _seed(db: SessionDB, n: int, prefix: str = "turn") -> list[int]:
    return [db.append_message("sess1", role="user" if i % 2 == 0 else "assistant", content=f"{prefix} {i}")
            for i in range(n)]


def _flags(db: SessionDB, content: str) -> list[tuple[int, int]]:
    rows = db._conn.execute(
        "SELECT active, compacted FROM messages WHERE session_id = 'sess1' AND content = ? ORDER BY id", (content,)
    ).fetchall()
    return [tuple(r) for r in rows]


def _live(db: SessionDB) -> list[str]:
    return [r["content"] for r in db.get_messages("sess1")]


SUMMARY = {"role": "user", "content": "[CONTEXT COMPACTION] summary"}


def test_only_the_held_rows_are_archived_and_a_gap_is_cloned_after_the_summary(db: SessionDB) -> None:
    held = _seed(db, 6)                        # rows 1-6, held
    gap = _seed(db, 2, prefix="foreign")       # rows 7-8, another surface's, never handed to the compressor
    own = _seed(db, 2, prefix="own")           # rows 9-10, held (the surface persisted them after the gap)
    watermark = db.get_active_message_watermark("sess1")
    assert watermark == own[-1]

    count = db.archive_and_compact("sess1", [dict(SUMMARY)], watermark=watermark, held_row_ids=[*held, *own])

    assert _live(db) == ["[CONTEXT COMPACTION] summary", "foreign 0", "foreign 1"]
    assert count == 3
    for i in range(6):
        assert _flags(db, f"turn {i}") == [(0, 1)]
    for i in range(2):
        assert _flags(db, f"own {i}") == [(0, 1)]
        assert _flags(db, f"foreign {i}") == [(0, 0), (1, 0)], "gap rows are cloned live, never summarized away"
    assert [r["id"] for r in db.get_messages("sess1")] == sorted(r["id"] for r in db.get_messages("sess1"))
    assert min(r["id"] for r in db.get_messages("sess1")) > gap[-1]


def test_rows_a_compacted_dict_still_names_are_rewound_not_archived(db: SessionDB) -> None:
    """The carried tail and a merged dict's sources are superseded duplicates (``active=0, compacted=0``), so
    search does not return each carried message once per compaction."""
    held = _seed(db, 6)
    merged_sources = [db.append_message("sess1", "user", "first half"), db.append_message("sess1", "user", "second half")]
    compacted = [
        dict(SUMMARY),
        {"role": "assistant", "content": "turn 5", "_row_id": held[5]},                       # verbatim tail copy
        {"role": "user", "content": "first half\n\nsecond half", "_row_ids": merged_sources},  # resume-merged dict
    ]

    db.archive_and_compact("sess1", compacted, held_row_ids=[*held, *merged_sources])

    assert _live(db) == ["[CONTEXT COMPACTION] summary", "turn 5", "first half\n\nsecond half"]
    for i in range(5):
        assert _flags(db, f"turn {i}") == [(0, 1)]
    assert _flags(db, "turn 5") == [(0, 0), (1, 0)]
    assert _flags(db, "first half") == [(0, 0)] and _flags(db, "second half") == [(0, 0)]
    assert all("_row_ids" not in m for m in compacted), "a fresh row stands for itself"
    assert [m["_row_id"] for m in compacted] == [r["id"] for r in db.get_messages("sess1")]


def test_a_held_row_that_is_no_longer_active_refuses_the_commit(db: SessionDB) -> None:
    held = _seed(db, 6)
    # Another surface compacted first: it archived 1-4 and cloned 5-6 (originals rewound), so none of the six
    # rows this surface still holds is active any more.
    db.archive_and_compact("sess1", [{"role": "user", "content": "winner summary"}], held_row_ids=held[:4])
    before = [(r["id"], r["content"]) for r in db.get_messages("sess1")]
    compacted = [{"role": "user", "content": "stale summary"}]

    with pytest.raises(StaleHeldRowsError) as excinfo:
        db.archive_and_compact("sess1", compacted, held_row_ids=held)

    assert excinfo.value.row_ids == held
    assert [(r["id"], r["content"]) for r in db.get_messages("sess1")] == before
    assert _flags(db, "stale summary") == [] and "_row_id" not in compacted[0]
    assert db.get_session("sess1")["message_count"] == len(before)


def test_without_held_row_ids_the_watermark_commit_is_unchanged(db: SessionDB) -> None:
    _seed(db, 6)
    watermark = db.get_active_message_watermark("sess1")
    db.append_message("sess1", "user", "arrived during the summary")

    db.archive_and_compact("sess1", [dict(SUMMARY)], watermark=watermark)

    assert _live(db) == ["[CONTEXT COMPACTION] summary", "arrived during the summary"]
    for i in range(6):
        assert _flags(db, f"turn {i}") == [(0, 1)]
