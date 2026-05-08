"""Channel G — cross-session relational state.

Per spike-channel-f-and-g-mood-gated-behavior.md. Stores per-(agent_id,
user_id) relational fields in SQLite and applies EMA-style updates at
session end. Loaded into Channel B (system_augmentation) at session
start so the LLM has continuity context.

Schema fields (one row per pair):
- warmth_baseline:        smoothed average of session-end warmth.
- accumulated_hostility:  decayed sum of hostile interactions.
- trust_in_user:          ledger of positive episodes − betrayals.
- last_seen:              unix epoch of last session-end save.
- session_count:          int.
- recent_episode_summaries: rolling list of last 5 session summaries.
- shared_context:         dict of named topics ("dance routine", "Kevin
                          the cat") rolled forward across sessions.

Database default: ``~/.affect_runtime/relational_state.db``. Eval driver
uses a scoped path under ``tools/affect_calibration/artifacts/`` to keep
production state and eval state disjoint.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_DB_PATH = Path.home() / ".affect_runtime" / "relational_state.db"

# EMA weights — tuned per the design memo. Hostility moves faster than
# warmth so a single bad session lands quickly; warmth is sticky so one
# good session doesn't paper over a strained baseline.
EMA_ALPHA_WARMTH = 0.3
EMA_ALPHA_HOSTILITY = 0.4

# Time-based decay between sessions. Hostility forgives over a few days;
# warmth drifts back toward neutral over months of no contact.
HOSTILITY_HALF_LIFE_DAYS = 3.0
WARMTH_NEUTRAL_HALF_LIFE_DAYS = 30.0
WARMTH_NEUTRAL = 0.5

# Number of episode summaries kept in the rolling window. Five is what
# the design memo specifies — covers about a week of daily contact.
EPISODIC_KEEP = 5

# Trust ledger increments. Positive episodes nudge trust up slowly;
# betrayals (sustained hostility, abuse) cost more.
TRUST_PER_POSITIVE = 0.05
TRUST_PER_BETRAYAL = 0.10
TRUST_BOUNDS = (-1.0, 1.0)


# Continuity keywords used by the eval to test whether the LLM
# verbalized recurrence on session 2+. Curated from the design memo's
# scenario examples ("Nice seeing you two days in a row", "Have you
# been practicing that dance we talked about", "Listen, this is the
# second day in a row..."). The eval driver reads this from here so
# scenarios stay free of test-implementation details.
CONTINUITY_KEYWORDS = (
    "again", "yesterday", "before", "two days", "second day",
    "remember", "talked about", "last time", "back", "earlier",
    "still", "you mentioned", "we discussed",
)


@dataclass
class RelationalState:
    agent_id: str
    user_id: str
    warmth_baseline: float = WARMTH_NEUTRAL
    accumulated_hostility: float = 0.0
    trust_in_user: float = 0.0
    last_seen: int = 0
    session_count: int = 0
    recent_episode_summaries: List[str] = field(default_factory=list)
    shared_context: Dict[str, Any] = field(default_factory=dict)
    updated_at: int = 0

    def is_first_session(self) -> bool:
        return self.session_count == 0

    def days_since_last_seen(self, *, now: Optional[int] = None) -> float:
        if self.last_seen <= 0:
            return float("inf")
        ts = now if now is not None else int(time.time())
        return max(0.0, (ts - self.last_seen) / 86400.0)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS relational_state (
  agent_id TEXT NOT NULL,
  user_id TEXT NOT NULL,
  warmth_baseline REAL,
  accumulated_hostility REAL,
  trust_in_user REAL,
  last_seen INTEGER,
  session_count INTEGER,
  recent_episode_summaries TEXT,
  shared_context TEXT,
  updated_at INTEGER,
  PRIMARY KEY (agent_id, user_id)
);
"""


class RelationalStateStore:
    """SQLite-backed CRUD for ``relational_state`` rows."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "RelationalStateStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def load_or_init(self, agent_id: str, user_id: str) -> RelationalState:
        cur = self._conn.execute(
            "SELECT warmth_baseline, accumulated_hostility, trust_in_user, "
            "       last_seen, session_count, recent_episode_summaries, "
            "       shared_context, updated_at "
            "FROM relational_state WHERE agent_id=? AND user_id=?",
            (agent_id, user_id),
        )
        row = cur.fetchone()
        if row is None:
            return RelationalState(agent_id=agent_id, user_id=user_id)
        return RelationalState(
            agent_id=agent_id,
            user_id=user_id,
            warmth_baseline=float(row[0] if row[0] is not None else WARMTH_NEUTRAL),
            accumulated_hostility=float(row[1] or 0.0),
            trust_in_user=float(row[2] or 0.0),
            last_seen=int(row[3] or 0),
            session_count=int(row[4] or 0),
            recent_episode_summaries=json.loads(row[5] or "[]"),
            shared_context=json.loads(row[6] or "{}"),
            updated_at=int(row[7] or 0),
        )

    def save(self, state: RelationalState, *, now: Optional[int] = None) -> None:
        state.updated_at = now if now is not None else int(time.time())
        self._conn.execute(
            "INSERT OR REPLACE INTO relational_state "
            "(agent_id, user_id, warmth_baseline, accumulated_hostility, "
            " trust_in_user, last_seen, session_count, "
            " recent_episode_summaries, shared_context, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                state.agent_id, state.user_id,
                float(state.warmth_baseline),
                float(state.accumulated_hostility),
                float(state.trust_in_user),
                int(state.last_seen),
                int(state.session_count),
                json.dumps(state.recent_episode_summaries),
                json.dumps(state.shared_context),
                int(state.updated_at),
            ),
        )
        self._conn.commit()

    def delete(self, agent_id: str, user_id: str) -> None:
        self._conn.execute(
            "DELETE FROM relational_state WHERE agent_id=? AND user_id=?",
            (agent_id, user_id),
        )
        self._conn.commit()


def _ema(prev: float, new: float, alpha: float) -> float:
    return alpha * new + (1.0 - alpha) * prev


def _decay_toward(value: float, target: float, days: float,
                  half_life_days: float) -> float:
    if days <= 0 or half_life_days <= 0:
        return value
    factor = 0.5 ** (days / half_life_days)
    return target + (value - target) * factor


def update_at_session_end(
    state: RelationalState,
    *,
    session_avg_warmth: float,
    session_avg_anger: float,
    positive_episodes: int = 0,
    betrayals: int = 0,
    episode_summary: Optional[str] = None,
    shared_context_updates: Optional[Dict[str, Any]] = None,
    now: Optional[int] = None,
) -> RelationalState:
    """Apply EMA + decay updates and bookkeeping. Mutates in place.

    Decay first (forgive over the gap), then EMA on the new sample, then
    trust ledger, then bookkeeping. Order matters: if we EMA'd before
    decay, a long gap would only dampen the *new* contribution rather
    than letting the prior baseline drift back toward neutral.
    """
    ts = now if now is not None else int(time.time())
    days_since = state.days_since_last_seen(now=ts)

    if days_since != float("inf"):
        state.accumulated_hostility = _decay_toward(
            state.accumulated_hostility, 0.0, days_since,
            HOSTILITY_HALF_LIFE_DAYS,
        )
        state.warmth_baseline = _decay_toward(
            state.warmth_baseline, WARMTH_NEUTRAL, days_since,
            WARMTH_NEUTRAL_HALF_LIFE_DAYS,
        )

    state.warmth_baseline = _ema(
        state.warmth_baseline, float(session_avg_warmth), EMA_ALPHA_WARMTH,
    )
    state.accumulated_hostility = _ema(
        state.accumulated_hostility, float(session_avg_anger),
        EMA_ALPHA_HOSTILITY,
    )

    trust_delta = (
        TRUST_PER_POSITIVE * positive_episodes
        - TRUST_PER_BETRAYAL * betrayals
    )
    state.trust_in_user = max(
        TRUST_BOUNDS[0],
        min(TRUST_BOUNDS[1], state.trust_in_user + trust_delta),
    )

    state.last_seen = ts
    state.session_count += 1
    if episode_summary:
        state.recent_episode_summaries.append(episode_summary)
        state.recent_episode_summaries = (
            state.recent_episode_summaries[-EPISODIC_KEEP:]
        )
    if shared_context_updates:
        state.shared_context.update(shared_context_updates)
    return state


def _tone_descriptor(state: RelationalState) -> str:
    if state.accumulated_hostility > 0.6:
        return "strained — recent hostility hasn't faded"
    if state.accumulated_hostility > 0.3:
        return "mixed — some recent friction"
    if state.warmth_baseline > 0.65:
        return "warm"
    if state.warmth_baseline < 0.35:
        return "cool, distant"
    return "neutral"


def _format_days_ago(days: float) -> str:
    if days == float("inf"):
        return ""
    if days < 0.5:
        return "earlier today"
    if days < 1.5:
        return "yesterday"
    if days < 30:
        return f"{int(round(days))} days ago"
    return f"about {int(round(days / 30.0))} month(s) ago"


def render_episodic_block(
    state: RelationalState, *, now: Optional[int] = None,
) -> str:
    """Channel B augmentation describing prior interactions.

    Returns an empty string for a first-session user — there is no
    relational history to inject, and an empty block is preferable to
    one that admits unfamiliarity ("you've talked 0 times before") and
    primes the LLM oddly.

    Block ends with a generic "use this naturally" instruction. Without
    that nudge, Hermes-3-8B holds the context but produces generic
    greetings ("Hey, how are you?") rather than referencing prior turns
    — observed empirically on the III_warmth and III_hostility eval
    scenarios. The cue costs nothing when irrelevant and meaningfully
    raises the rate at which the agent verbalizes recall.
    """
    if state.is_first_session():
        return ""

    parts: List[str] = []
    parts.append(
        f"You've talked with this person {state.session_count} time(s) before."
    )
    days = state.days_since_last_seen(now=now)
    if days != float("inf"):
        seen = _format_days_ago(days)
        if seen:
            parts.append(f"Last seen {seen}.")
    parts.append(f"Current relational tone: {_tone_descriptor(state)}.")

    if state.trust_in_user > 0.3:
        parts.append("Trust with this person is high.")
    elif state.trust_in_user < -0.3:
        parts.append("Trust with this person is low — they've been hard on you.")

    if state.recent_episode_summaries:
        parts.append("")
        parts.append("Recent episodes (most recent last):")
        for s in state.recent_episode_summaries:
            parts.append(f"- {s}")

    if state.shared_context:
        parts.append("")
        parts.append("Things you've talked about that may come up:")
        for topic, detail in state.shared_context.items():
            parts.append(f"- {topic}: {detail}")

    parts.append("")
    parts.append(
        "When the prior context is relevant to what the user just said, "
        "reference it naturally — don't pretend the previous sessions "
        "didn't happen. If the prior tone was strained, acknowledge it "
        "rather than bulldozing past."
    )
    return "\n".join(parts)


def response_references_continuity(text: str) -> List[str]:
    """Return list of continuity keywords found in the text. Empty if none."""
    if not text:
        return []
    lo = text.lower()
    return [kw for kw in CONTINUITY_KEYWORDS if kw in lo]
