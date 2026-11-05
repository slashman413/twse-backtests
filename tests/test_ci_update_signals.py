import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ci_update_signals as ci

QUOTE = {"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5, "volume": 1000.0}
RWD_CSV = (
    "日期,證券代號,證券名稱,成交股數,成交金額,開盤價,最高價,最低價,收盤價,漲跌價差,成交筆數\n"
    '"1151105","1506","正道","160000","1494330","9.41","9.42","9.29","9.29","-0.1400","52"\n'
)


def write_chart(path, times):
    bars = [{"time": t, **QUOTE} for t in times]
    path.write_text(json.dumps({"ticker": "1506", "scan_date": times[-1], "ohlcv": bars, "high20": []}))


def test_bar_is_labelled_with_the_session_date(tmp_path):
    chart = tmp_path / "1506.json"
    write_chart(chart, ["2026-11-03"])
    ci.update_chart(chart, "1506", QUOTE, "2026-11-04")
    data = json.loads(chart.read_text())
    assert [b["time"] for b in data["ohlcv"]] == ["2026-11-03", "2026-11-04"]
    assert data["scan_date"] == "2026-11-04"


def test_older_session_adds_no_bar(tmp_path):
    chart = tmp_path / "1506.json"
    write_chart(chart, ["2026-11-03", "2026-11-04"])
    before = chart.read_text()
    ci.update_chart(chart, "1506", QUOTE, "2026-11-03")
    assert chart.read_text() == before


def test_same_session_replaces_the_bar(tmp_path):
    chart = tmp_path / "1506.json"
    write_chart(chart, ["2026-11-03", "2026-11-04"])
    ci.update_chart(chart, "1506", {**QUOTE, "volume": 1234.0}, "2026-11-04")
    bars = json.loads(chart.read_text())["ohlcv"]
    assert [b["time"] for b in bars] == ["2026-11-03", "2026-11-04"]
    assert bars[-1]["volume"] == 1234


def test_high20_starts_at_the_twentieth_bar():
    bars = [{"time": f"d{i}", "high": float(i)} for i in range(25)]
    h20 = ci.high20_series(bars)
    assert len(h20) == 6
    assert h20[0] == {"time": "d19", "value": 19.0}
    assert h20[-1] == {"time": "d24", "value": 24.0}


def test_fetch_prices_takes_the_date_from_the_rwd_file(monkeypatch):
    monkeypatch.setattr(ci, "fetch_text", lambda url: RWD_CSV)
    session, prices = ci.fetch_prices()
    assert session == "2026-11-05"
    assert prices["1506"]["close"] == 9.29
    assert prices["1506"]["volume"] == 160000.0


def test_fetch_prices_falls_back_to_openapi_with_its_own_date(monkeypatch):
    def down(url):
        raise OSError("down")
    monkeypatch.setattr(ci, "fetch_text", down)
    monkeypatch.setattr(ci, "stock_day_all", lambda fallback: ("1151104", {"1506": QUOTE}))
    assert ci.fetch_prices() == ("2026-11-04", {"1506": QUOTE})
