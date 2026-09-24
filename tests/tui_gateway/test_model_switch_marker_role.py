"""Tests for _append_model_switch_marker role fix (issue #48338).

The model switch marker must NOT use role="system" because strict providers
(vLLM, Qwen) reject system messages that appear mid-conversation. Using
role="user" is safe — the system prompt is prepended to the API message list,
so a user-role marker can appear at any later position, and the gateway's
sanitize/merge pass already coalesces consecutive user messages.
"""

from __future__ import annotations


from tui_gateway.server import _append_model_switch_marker


class TestAppendModelSwitchMarkerRole:
    """Verify the marker uses role='user', not role='system'."""

    def test_marker_uses_user_role(self) -> None:
        """The history entry must be role='user', not role='system'."""
        session: dict = {"session_key": "test-session", "history": []}
        _append_model_switch_marker(session, model="gpt-4o", provider="openai")
        assert len(session["history"]) == 1
        entry = session["history"][0]
        assert entry["role"] == "user", (
            f"Expected role='user' but got role='{entry['role']}'. "
            "Strict providers (vLLM, Qwen) reject mid-conversation system messages."
        )




class TestModelSwitchMarkerDedup:
    """#65891: only the newest marker is meaningful; older ones must not
    accumulate in the live history and burn context tokens every turn."""

    @staticmethod
    def _markers(session: dict) -> list:
        from tui_gateway.server import _is_model_switch_marker

        return [h for h in session["history"] if _is_model_switch_marker(h)]

    def test_second_switch_replaces_first_marker(self) -> None:
        session: dict = {"session_key": "s", "history": []}
        _append_model_switch_marker(session, model="model-a", provider="p")
        _append_model_switch_marker(session, model="model-b", provider="p")
        markers = self._markers(session)
        assert len(markers) == 1, "a second switch must replace, not stack, the marker"
        assert "model-b" in markers[0]["content"]
        assert "model-a" not in markers[0]["content"]
        # The surviving marker is the last history entry.
        assert session["history"][-1] is markers[0]


    def test_dedup_preserves_real_conversation_turns(self) -> None:
        session: dict = {
            "session_key": "s",
            "history": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
        }
        _append_model_switch_marker(session, model="model-a", provider="p")
        _append_model_switch_marker(session, model="model-b", provider="p")
        # Real turns untouched; exactly one marker, appended at the end.
        assert session["history"][0] == {"role": "user", "content": "hello"}
        assert session["history"][1] == {"role": "assistant", "content": "hi"}
        assert len(self._markers(session)) == 1
        assert len(session["history"]) == 3

    def test_prior_marker_between_turns_is_removed(self) -> None:
        # A stale marker not at the tail (a later turn followed it) is still
        # stripped on the next switch.
        session: dict = {
            "session_key": "s",
            "history": [
                {"role": "user", "content": "q1"},
                _make_marker_entry("model-a"),
                {"role": "assistant", "content": "a1"},
            ],
        }
        _append_model_switch_marker(session, model="model-b", provider="p")
        markers = self._markers(session)
        assert len(markers) == 1
        assert "model-b" in markers[0]["content"]
        # The real turns are preserved in order.
        assert [h["content"] for h in session["history"] if not _is_marker(h)] == ["q1", "a1"]

    def test_history_version_increments_once_on_replace(self) -> None:
        session: dict = {"session_key": "s", "history": [], "history_version": 0}
        _append_model_switch_marker(session, model="model-a", provider="p")
        _append_model_switch_marker(session, model="model-b", provider="p")
        assert session["history_version"] == 2  # one increment per switch


class TestModelSwitchMarkerIsStampedLikeAFlushedRow:
    """#121734: the marker is a durable row the surface holds, so its live dict must name its row id and carry
    the persist marker. The in-place compaction commit archives by the ids the held history names; an unstamped
    durable row looks like an unpersisted turn and would be cloned back beside its own copy."""

    def test_entry_names_its_durable_row(self, tmp_path) -> None:
        from agent.context_compressor import _DB_PERSISTED_MARKER
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            db.create_session("s", "tui", model="m")
            agent = type("Agent", (), {"_session_db": db})()
            session: dict = {"session_key": "s", "history": [], "agent": agent}
            _append_model_switch_marker(session, model="model-a", provider="p")
            entry = session["history"][-1]
            rows = db.get_messages_as_conversation("s", include_row_ids=True)
            assert len(rows) == 1 and rows[0]["display_kind"] == "model_switch"
            assert entry["_row_id"] == rows[0]["_row_id"] and entry[_DB_PERSISTED_MARKER] is True
        finally:
            db.close()

    def test_entry_stays_unstamped_when_the_write_failed(self) -> None:
        class _Db:
            def append_message(self, **_kwargs):
                raise RuntimeError("disk full")

        session: dict = {"session_key": "s", "history": [], "agent": type("Agent", (), {"_session_db": _Db()})()}
        _append_model_switch_marker(session, model="model-a", provider="p")
        assert "_row_id" not in session["history"][-1]


def _make_marker_entry(model: str) -> dict:
    from tui_gateway.server import _MODEL_SWITCH_MARKER_PREFIX

    return {"role": "user", "content": f"{_MODEL_SWITCH_MARKER_PREFIX}{model}.]"}


def _is_marker(entry: dict) -> bool:
    from tui_gateway.server import _is_model_switch_marker

    return _is_model_switch_marker(entry)
