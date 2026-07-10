"""Global, always-on intraday concept-board anomaly scan.

Runs as an independent asyncio task inside the scheduler process (see
scheduler/main.py), separate from the per-user scheduled-analysis loop.
Trading days only, 9:30-15:00 (lunch break skipped), 5-minute polling
that boosts to 2 minutes for 15 minutes after any anomaly is found.

Detection is pure code (scheduler/intraday_rules.py); the only LLM call
is the narrow cause-attribution agent (scheduler/intraday_agent.py), and
only for boards that actually tripped a rule. See
docs/superpowers/specs/2026-07-10-intraday-analysis-design.md.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from uuid import uuid4
from zoneinfo import ZoneInfo

from scheduler.intraday_agent import analyze_cause
from scheduler.intraday_rules import BoardAnomaly, detect_board_anomalies, detect_zt_pool_concentration

logger = logging.getLogger(__name__)

_NORMAL_INTERVAL_SECONDS = 300  # 5 min
_BOOST_INTERVAL_SECONDS = 120  # 2 min
_BOOST_DURATION_SECONDS = 900  # 15 min

_TZ = ZoneInfo("Asia/Shanghai")


class _BoostState:
    """Tracks the temporary poll-interval speedup after an anomaly is found."""

    def __init__(self) -> None:
        self._boosted_until_ts = 0.0

    def trigger(self, now_ts: float) -> None:
        self._boosted_until_ts = now_ts + _BOOST_DURATION_SECONDS

    def current_interval(self, now_ts: float) -> int:
        return _BOOST_INTERVAL_SECONDS if now_ts < self._boosted_until_ts else _NORMAL_INTERVAL_SECONDS


def _within_trading_window(now: datetime) -> bool:
    hhmm = now.hour * 60 + now.minute
    if not (9 * 60 + 30 <= hhmm <= 15 * 60):
        return False
    if 11 * 60 + 30 <= hhmm < 13 * 60:
        return False  # lunch break
    return True


def _persist_signal(trade_date: str, anomaly: BoardAnomaly, result) -> None:
    from api.database import IntradaySignalDB, get_db_ctx

    with get_db_ctx() as db:
        db.add(
            IntradaySignalDB(
                id=uuid4().hex,
                trade_date=trade_date,
                board_name=anomaly.board_name,
                anomaly_case=anomaly.anomaly_case,
                change_pct=anomaly.change_pct,
                net_inflow=anomaly.net_inflow,
                cause_summary=result.cause_summary,
                fund_source=result.fund_source,
                judgement=result.judgement,
                llm_failed=result.llm_failed,
            )
        )
        db.commit()


async def run_one_tick(trade_date: str) -> bool:
    """Fetch → detect → attribute → persist. Returns True iff any anomaly was found (drives boost)."""
    from tradingagents.dataflows.providers.cn_akshare_provider import CnAkshareProvider

    provider = CnAkshareProvider()
    concept_df = await asyncio.to_thread(provider.get_concept_fund_flow_df)
    zt_df = await asyncio.to_thread(provider.get_zt_pool_df, trade_date)

    board_anomalies = detect_board_anomalies(concept_df)
    zt_concentrated = detect_zt_pool_concentration(zt_df)

    if not board_anomalies and not zt_concentrated:
        return False

    if board_anomalies:
        from api.main import _build_runtime_config

        config = await asyncio.to_thread(_build_runtime_config, {})
        for anomaly in board_anomalies:
            result = await analyze_cause(anomaly, trade_date, config)
            await asyncio.to_thread(_persist_signal, trade_date, anomaly, result)
            logger.info(
                "[Intraday] %s Case %s persisted (llm_failed=%s)",
                anomaly.board_name, anomaly.anomaly_case, result.llm_failed,
            )

    return True


async def intraday_loop() -> None:
    """Background loop: trading-hours-only concept-board anomaly scan. Runs forever."""
    from tradingagents.dataflows.trade_calendar import is_cn_trading_day

    boost = _BoostState()
    logger.info("[Intraday] Loop started.")
    while True:
        now = datetime.now(tz=_TZ)
        interval = _NORMAL_INTERVAL_SECONDS
        try:
            today = now.strftime("%Y-%m-%d")
            if is_cn_trading_day(today) and _within_trading_window(now):
                found = await run_one_tick(today)
                if found:
                    boost.trigger(now.timestamp())
                    logger.info("[Intraday] anomaly found, boosting to %ds interval", _BOOST_INTERVAL_SECONDS)
        except Exception as exc:
            # A single bad tick (akshare hiccup, DB blip, etc.) must never kill the loop.
            logger.error("[Intraday] tick failed: %s", exc)
        interval = boost.current_interval(datetime.now(tz=_TZ).timestamp())
        await asyncio.sleep(interval)
