from datetime import datetime
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from scheduler.intraday import _BoostState, _within_trading_window, run_one_tick
from scheduler.intraday_rules import BoardAnomaly


def _dt(hh, mm):
    return datetime(2026, 7, 10, hh, mm)


@pytest.mark.parametrize(
    "hh,mm,expected",
    [
        (9, 29, False),   # before open
        (9, 30, True),    # right at open
        (11, 29, True),   # just before lunch
        (11, 30, False),  # lunch starts
        (12, 30, False),  # lunch
        (12, 59, False),  # still lunch
        (13, 0, True),    # lunch ends
        (15, 0, True),    # right at close
        (15, 1, False),   # after close
    ],
)
def test_within_trading_window(hh, mm, expected):
    assert _within_trading_window(_dt(hh, mm)) is expected


def test_boost_state_default_is_normal_interval():
    boost = _BoostState()
    assert boost.current_interval(1000.0) == 300


def test_boost_state_triggers_then_expires():
    boost = _BoostState()
    boost.trigger(now_ts=1000.0)
    assert boost.current_interval(1000.0) == 120
    assert boost.current_interval(1000.0 + 900 - 1) == 120
    assert boost.current_interval(1000.0 + 900 + 1) == 300


@pytest.mark.asyncio
async def test_run_one_tick_no_anomaly_does_nothing():
    with patch("tradingagents.dataflows.providers.cn_akshare_provider.CnAkshareProvider.get_concept_fund_flow_df", return_value=pd.DataFrame()), \
         patch("tradingagents.dataflows.providers.cn_akshare_provider.CnAkshareProvider.get_zt_pool_df", return_value=pd.DataFrame()), \
         patch("scheduler.intraday.analyze_cause", new=AsyncMock()) as mock_analyze, \
         patch("scheduler.intraday._persist_signal") as mock_persist:
        found = await run_one_tick("2026-07-10")
    assert found is False
    mock_analyze.assert_not_called()
    mock_persist.assert_not_called()


@pytest.mark.asyncio
async def test_run_one_tick_zt_concentration_only_boosts_without_llm_call():
    zt_df = pd.DataFrame([{"code": i} for i in range(31)])
    with patch("tradingagents.dataflows.providers.cn_akshare_provider.CnAkshareProvider.get_concept_fund_flow_df", return_value=pd.DataFrame()), \
         patch("tradingagents.dataflows.providers.cn_akshare_provider.CnAkshareProvider.get_zt_pool_df", return_value=zt_df), \
         patch("scheduler.intraday.analyze_cause", new=AsyncMock()) as mock_analyze, \
         patch("scheduler.intraday._persist_signal") as mock_persist:
        found = await run_one_tick("2026-07-10")
    assert found is True
    mock_analyze.assert_not_called()
    mock_persist.assert_not_called()


@pytest.mark.asyncio
async def test_run_one_tick_board_anomaly_triggers_attribution_and_persist():
    concept_df = pd.DataFrame([{"行业": "CRO概念", "行业-涨跌幅": 4.43, "净额": 19.56}])
    fake_result = type("R", (), {"cause_summary": "text", "fund_source": "机构", "judgement": "持续行情", "llm_failed": False})()

    with patch("tradingagents.dataflows.providers.cn_akshare_provider.CnAkshareProvider.get_concept_fund_flow_df", return_value=concept_df), \
         patch("tradingagents.dataflows.providers.cn_akshare_provider.CnAkshareProvider.get_zt_pool_df", return_value=pd.DataFrame()), \
         patch("scheduler.intraday.analyze_cause", new=AsyncMock(return_value=fake_result)) as mock_analyze, \
         patch("scheduler.intraday._persist_signal") as mock_persist, \
         patch("api.main._build_runtime_config", return_value={"llm_provider": "openai"}):
        found = await run_one_tick("2026-07-10")

    assert found is True
    mock_analyze.assert_called_once()
    called_anomaly = mock_analyze.call_args[0][0]
    assert isinstance(called_anomaly, BoardAnomaly)
    assert called_anomaly.board_name == "CRO概念"
    mock_persist.assert_called_once_with("2026-07-10", called_anomaly, fake_result)
