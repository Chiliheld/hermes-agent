"""Internal metadata attached to durable conversation messages."""

from __future__ import annotations

from time import time as wall_time
from typing import Any, MutableMapping, Optional, TypeVar


# These fields describe Hermes' durable record and timeline display, not
# provider-visible message content. The request builder strips them from every
# outgoing copy and the token estimator ignores them: one set, so an estimate
# never prices bytes the provider never receives (an edit's inline_diff in
# display_metadata is ~9KB and would trigger premature compaction).
PERSISTENCE_ONLY_MESSAGE_FIELDS = frozenset({"timestamp", "display_kind", "display_metadata", "_row_id", "_row_ids"})

_Message = TypeVar("_Message", bound=MutableMapping[str, Any])


def message_row_ids(message: Any) -> list[int]:
    """Every durable row id a live message names, in order and without repeats.

    ``_row_id`` is the row the dict was loaded from or flushed as. ``_row_ids`` is merge coverage: a pass
    that folded another message into this dict (``repair_message_sequence``'s user/assistant merges, the
    ``here N`` seam fold) records every source row there, so an in-place compaction that summarized the
    merged dict archives all the rows it stands for, not only the surviving id. Non-positive, boolean and
    non-integer values are ignored.
    """
    if not isinstance(message, dict):
        return []
    raw = message.get("_row_ids")
    candidates = [message.get("_row_id"), *(raw if isinstance(raw, (list, tuple)) else ())]
    return list(dict.fromkeys(
        int(rid) for rid in candidates if isinstance(rid, int) and not isinstance(rid, bool) and rid > 0
    ))


def record_merge_coverage(survivor: MutableMapping[str, Any], absorbed: Any) -> None:
    """Make *survivor* name every durable row it now covers after *absorbed* was folded into it.

    ``_row_id`` stays the survivor's own row (flush and reaction paths address that one); ``_row_ids`` gets
    the union of both sides' ids. Set only when there is something to record, so dicts that never touched
    the DB stay free of persistence-only keys.
    """
    ids = list(dict.fromkeys([*message_row_ids(survivor), *message_row_ids(absorbed)]))
    if ids:
        survivor["_row_ids"] = ids


def stamp_message_timestamp(
    message: _Message,
    *,
    timestamp: Optional[float] = None,
) -> _Message:
    """Attach a creation timestamp without replacing source-provided time.

    Gateway adapters can supply the platform event time; all other callers use
    the local wall clock. Returns the same mapping for use at append sites.
    """
    if message.get("timestamp") is None:
        message["timestamp"] = wall_time() if timestamp is None else timestamp
    return message


def append_message(
    messages: list[Any],
    message: _Message,
    *,
    timestamp: Optional[float] = None,
) -> _Message:
    """Stamp and append one live transcript message."""
    messages.append(stamp_message_timestamp(message, timestamp=timestamp))
    return message
