"""
TWSE 量化交易掃描器 — 每日掃描與排程
======================================

功能：
    1. 全市場上市股票掃描 (買/賣訊號)
    2. 大盤多空判定
    3. 類股燈號
    4. 可讀報表 (文字 + HTML)
    5. Cron job 排程整合

使用方式：
    python main.py                         # 掃描預設權值股
    python main.py --all                   # 掃描全市場 (約 1000 檔)
    python main.py --watchlist 2330,2454   # 掃描指定清單
    python main.py --report report.html    # 輸出 HTML 報表
"""

from __future__ import annotations

import argparse
import html as _html
import os
import sys
import time
from datetime import date, datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

from data_loader import TWSEStockLoader
from indicators import macd, dmi, rsi, vr, wr, n2, k6k9
from strategy import (
    BigStockBuySignal,
    BigStockSellSignal,
    CrashExitSignal,
    EntrySignal,
    MarketAssessment,
    MarketSignal,
    MarketSignalV2,
    MarketState,
    ScanReport,
    SectorLight,
    SectorSignal,
    SignalAligner,
    SignalResult,
    StockSurgeSignal,
    TradeSignal,
)

# ── 預設掃描清單 ──────────────────────────────────────────────

DEFAULT_WATCHLIST = [
    "2330", "2454", "2317", "2308", "2412",  # 權值電子
    "2002", "1301", "1303", "1326",  # 傳產權值
    "2881", "2882", "2886", "2891", "5880",  # 金融
    "3008", "3711", "8046", "2357", "2382",  # 高價電子
    "1101", "1216", "1402", "1504", "2105",  # 各類股龍頭
    "2603", "2615", "2618",  # 航運
    "3034", "3231", "2379", "2301", "2327",  # 電子中型
]

# 台灣50 成分股 (權值代表性)
TW50 = [
    "2330", "2454", "2317", "2308", "2412",  # 電子
    "2881", "2882", "2886", "2891", "5880",  # 金融
    "2002", "1301", "1303", "1326",  # 傳產
    "3008", "3711", "8046", "2357", "2382",  # 高價
    "1101", "1216", "1402", "1504",  # 各類
    "2603", "3034", "3231", "2379", "2301", "2327",  # 電子
    "4904", "4938", "5347", "6239", "6269",
    "6446", "6669", "6732", "6742", "6770",
    "8016", "8299", "8454",
]

# ── 路徑 ──────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(PROJECT_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# 1. 每日掃描器
# ═══════════════════════════════════════════════════════════════

class DailyScanner:
    """每日掃描器 — 全市場 / 指定清單掃描.

    流程：
        1. 載入大盤資料，判定多空
        2. 載入類股資料，判定燈號
        3. 逐檔掃描買進/賣出訊號
        4. 產出報表
    """

    def __init__(
        self,
        loader: TWSEStockLoader | None = None,
        *,
        batch_size: int = 20,
        max_workers: int = 1,
    ):
        self.loader = loader or TWSEStockLoader()
        self.batch_size = batch_size
        self.buy_signal = BigStockBuySignal(self.loader)
        self.sell_signal = BigStockSellSignal(self.loader)
        self.market_signal = MarketSignal(self.loader)
        self.sector_signal = SectorSignal(self.loader)
        self.aligner = SignalAligner(self.loader)

    def scan(
        self,
        tickers: list[str] | None = None,
        *,
        progress: bool = True,
        align_signals: bool = True,
    ) -> ScanReport:
        """執行一次完整掃描.

        Args:
            tickers: 掃描清單 (None=預設權值清單)
            progress: 是否顯示進度
            align_signals: 是否執行跨週期對齊

        Returns:
            ScanReport
        """
        tickers = tickers or DEFAULT_WATCHLIST
        report = ScanReport()
        report.date = date.today().isoformat()

        # ── 1. 大盤判定 ──
        if progress:
            print("\n📊 大盤多空判定...")
        try:
            market_df = self.loader.load_daily(
                MarketSignal.MARKET_TICKER, adjusted=True,
            )
            report.market = self.market_signal.assess(market_df)
        except Exception as e:
            report.market = MarketAssessment(reasons=[f"大盤讀取失敗: {e}"])

        # ── 2. 類股燈號 ──
        if progress:
            print("📊 類股燈號判定...")
        try:
            if market_df is not None and not market_df.empty:
                report.sector_signals = self.sector_signal.assess_all(market_df)
        except Exception:
            pass

        # ── 3. 逐檔掃描 ──
        if progress:
            print(f"\n🔍 掃描 {len(tickers)} 檔股票...")

        buy_list: list[SignalResult] = []
        sell_list: list[SignalResult] = []

        for i, ticker in enumerate(tickers):
            if progress:
                self._show_progress(i, len(tickers), ticker)

            try:
                # 買進訊號
                buy_result = self.buy_signal.evaluate(ticker)

                # 賣出訊號
                sell_result = self.sell_signal.evaluate(ticker)

                # 跨週期對齊 (只對買進訊號)
                if align_signals and buy_result.signal in (
                    TradeSignal.BUY, TradeSignal.STRONG_BUY,
                ):
                    tf = self.loader.load_multi_timeframe(ticker, adjusted=True)
                    buy_result = self.aligner.align(
                        ticker, buy_result,
                        daily=tf["daily"],
                        weekly=tf["weekly"],
                        monthly=tf["monthly"],
                    )

                # 收集
                if buy_result.signal in (
                    TradeSignal.BUY, TradeSignal.STRONG_BUY,
                ):
                    buy_list.append(buy_result)
                elif sell_result.signal in (
                    TradeSignal.SELL, TradeSignal.STRONG_SELL,
                ):
                    sell_list.append(sell_result)

            except Exception as e:
                if progress:
                    print(f"\n    ⚠️ {ticker}: {e}")

        if progress:
            self._show_progress(len(tickers), len(tickers), "完成")
            print()

        report.buy_list = sorted(
            buy_list, key=lambda r: r.confidence, reverse=True,
        )
        report.sell_list = sorted(
            sell_list, key=lambda r: r.confidence, reverse=True,
        )
        report.summary = self._build_summary(report)

        return report

    @staticmethod
    def _show_progress(current: int, total: int, ticker: str):
        """顯示掃描進度條."""
        pct = current / total * 100 if total > 0 else 0
        bar_len = 20
        filled = int(bar_len * current / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r  [{bar}] {current}/{total} ({pct:.0f}%) {ticker}", end="")
        sys.stdout.flush()

    @staticmethod
    def _build_summary(report: ScanReport) -> str:
        """建立掃描摘要文字."""
        lines: list[str] = []
        lines.append(f"📅 掃描日期: {report.date}")
        lines.append(f"📊 大盤狀態: {report.market.state.value} "
                      f"(score={report.market.score})")
        lines.append(f"📈 建議買進: {len(report.buy_list)} 檔")
        lines.append(f"📉 建議減碼: {len(report.sell_list)} 檔")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 2. 報表產生器
# ═══════════════════════════════════════════════════════════════

class ReportGenerator:
    """報表產生器 — 文字 / HTML / CSV."""

    @staticmethod
    def text_report(report: ScanReport) -> str:
        """產生純文字報表."""
        lines: list[str] = []
        sep = "=" * 60

        # ── 標頭 ──
        lines.append(sep)
        lines.append(f"  TWSE 量化掃描報表 — {report.date}")
        lines.append(sep)
        lines.append("")

        # ── 大盤狀態 ──
        lines.append("📊 大盤狀態")
        lines.append("-" * 40)
        m = report.market
        lines.append(f"  判定: {m.state.value} (score={m.score})")
        for r in m.reasons:
            lines.append(f"    • {r}")
        if m.indicators:
            lines.append(f"  指標: {m.indicators}")
        lines.append("")

        # ── 類股燈號 ──
        if report.sector_signals:
            lines.append("📊 類股燈號")
            lines.append("-" * 40)
            for sector, light in sorted(report.sector_signals.items()):
                lines.append(f"  {light.value} {sector}")
            lines.append("")

        # ── 買進清單 ──
        lines.append(f"📈 建議買進 ({len(report.buy_list)} 檔)")
        lines.append("-" * 40)
        if report.buy_list:
            for r in report.buy_list[:20]:  # 最多顯示 20 檔
                met = len(r.conditions_met)
                total = met + len(r.conditions_failed)
                icon = "🟢" if r.signal == TradeSignal.STRONG_BUY else "🟡"
                lines.append(
                    f"  {icon} {r.ticker}: {r.signal.value} "
                    f"({met}/{total}, {r.confidence:.0%})"
                )
                for c in r.conditions_met[:3]:
                    lines.append(f"      ✅ {c}")
                if r.detail.get("alignment"):
                    lines.append(f"      📐 {r.detail['alignment']}")
        else:
            lines.append("  (無符合條件標的)")
        lines.append("")

        # ── 賣出清單 ──
        lines.append(f"📉 建議減碼/賣出 ({len(report.sell_list)} 檔)")
        lines.append("-" * 40)
        if report.sell_list:
            for r in report.sell_list[:10]:
                icon = "🔴" if r.signal == TradeSignal.STRONG_SELL else "🟠"
                lines.append(
                    f"  {icon} {r.ticker}: {r.signal.value} "
                    f"({r.confidence:.0%})"
                )
                for c in r.conditions_met[:3]:
                    lines.append(f"      ⚠️ {c}")
        else:
            lines.append("  (無符合條件標的)")
        lines.append("")

        # ── 摘要 ──
        lines.append(sep)
        lines.append(report.summary)
        lines.append(sep)

        return "\n".join(lines)

    @staticmethod
    def html_report(report: ScanReport, output_path: str | None = None) -> str:
        """產生 HTML 報表."""
        m = report.market

        state_colors = {
            MarketState.BULL: "#22c55e",
            MarketState.ALERT: "#eab308",
            MarketState.BEAR: "#ef4444",
            MarketState.CRASH: "#dc2626",
        }
        state_color = state_colors.get(m.state, "#6b7280")

        buy_rows = ""
        for r in report.buy_list[:30]:
            met = len(r.conditions_met)
            total = met + len(r.conditions_failed)
            pct = int(r.confidence * 100)
            color = "#22c55e" if r.signal == TradeSignal.STRONG_BUY else "#eab308"
            conds = "<br>".join(
                [f"✅ {c}" for c in r.conditions_met[:4]]
            )
            align = r.detail.get("alignment", "")
            align_html = f"<br><small>📐 {align}</small>" if align else ""
            buy_rows += f"""
            <tr>
                <td style="font-weight:bold;color:{color}">{_html.escape(r.ticker)}</td>
                <td>{_html.escape(r.signal.value)}</td>
                <td>{met}/{total}</td>
                <td>
                    <div style="background:#e5e7eb;border-radius:8px;height:8px;width:100px">
                        <div style="background:{color};border-radius:8px;height:8px;width:{pct}px"></div>
                    </div>
                </td>
                <td style="font-size:12px">{conds}{align_html}</td>
            </tr>"""

        sell_rows = ""
        for r in report.sell_list[:15]:
            color = "#dc2626" if r.signal == TradeSignal.STRONG_SELL else "#f97316"
            conds = "<br>".join(
                [f"⚠️ {c}" for c in r.conditions_met[:3]]
            )
            sell_rows += f"""
            <tr>
                <td style="font-weight:bold;color:{color}">{_html.escape(r.ticker)}</td>
                <td>{_html.escape(r.signal.value)}</td>
                <td style="font-size:12px">{conds}</td>
            </tr>"""

        sector_rows = ""
        if report.sector_signals:
            for sector, light in sorted(report.sector_signals.items()):
                emoji = {"🟢": "#22c55e", "🟡": "#eab308", "🔴": "#ef4444"}.get(
                    light.value[:2], "#6b7280"
                )
                sector_rows += f"""
                <tr>
                    <td>{sector}</td>
                    <td style="color:{emoji};font-weight:bold">{light.value}</td>
                </tr>"""

        html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head><meta charset="utf-8">
<title>TWSE 量化掃描報表 {report.date}</title>
<style>
body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #f8fafc; color: #1e293b; }}
h1 {{ font-size: 20px; color: #0f172a; }}
h2 {{ font-size: 16px; margin-top: 24px; }}
table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
th {{ text-align: left; padding: 10px 12px; background: #f1f5f9; font-size: 12px; text-transform: uppercase; }}
td {{ padding: 8px 12px; border-top: 1px solid #f1f5f9; font-size: 13px; }}
.market-box {{ padding: 16px; border-radius: 8px; color: white; font-weight: bold; font-size: 18px; margin: 8px 0; }}
.summary {{ background: white; padding: 12px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin: 16px 0; line-height: 1.6; }}
</style>
</head><body>
<h1>📊 TWSE 量化掃描報表</h1>
<p style="color:#64748b">{report.date}</p>

<div class="market-box" style="background:{state_color}">
    大盤狀態: {m.state.value} (score={m.score})
</div>
<div class="summary">
    <b>多空理由:</b><br>
    {"<br>".join(f"• {r}" for r in m.reasons[:5])}
</div>

<h2>📊 類股燈號</h2>
<table>
<tr><th>類股</th><th>燈號</th></tr>
{sector_rows or "<tr><td colspan=2>無資料</td></tr>"}
</table>

<h2>📈 建議買進 ({len(report.buy_list)} 檔)</h2>
<table>
<tr><th>代號</th><th>訊號</th><th>條件</th><th>信心</th><th>細節</th></tr>
{buy_rows or "<tr><td colspan=5 style='text-align:center;color:#94a3b8'>無符合條件標的</td></tr>"}
</table>

<h2>📉 建議減碼/賣出 ({len(report.sell_list)} 檔)</h2>
<table>
<tr><th>代號</th><th>訊號</th><th>細節</th></tr>
{sell_rows or "<tr><td colspan=3 style='text-align:center;color:#94a3b8'>無符合條件標的</td></tr>"}
</table>

<p style="color:#64748b;background:#f1f5f9;border-radius:8px;padding:12px 16px;margin-top:28px;font-size:12px;line-height:1.6">
    ⚠️ 免責聲明：本報表由程式依公開量化規則自動產生，「建議買進 / 減碼 / 賣出」等標示為<strong>依規則計算之研究資訊，非投資建議或買賣推薦</strong>。回測與訊號為假設情境，過去或模擬績效不代表未來結果。台股投資有風險，可能損失本金，請自行判斷並承擔風險。
</p>
<p style="color:#94a3b8;text-align:center;margin-top:16px;font-size:12px">
    Generated by Hermes TWSE Quant Screener • {datetime.now().strftime("%Y-%m-%d %H:%M")}
</p>
</body></html>"""

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html)

        return html

    @staticmethod
    def csv_report(report: ScanReport, output_path: str | None = None) -> str:
        """產出 CSV 格式報表."""
        rows = []
        for r in report.buy_list:
            rows.append({
                "Ticker": r.ticker, "Signal": r.signal.value,
                "Type": "BUY", "Confidence": f"{r.confidence:.0%}",
                "ConditionsMet": len(r.conditions_met),
                "TotalConditions": len(r.conditions_met) + len(r.conditions_failed),
            })
        for r in report.sell_list:
            rows.append({
                "Ticker": r.ticker, "Signal": r.signal.value,
                "Type": "SELL", "Confidence": f"{r.confidence:.0%}",
                "ConditionsMet": len(r.conditions_met),
                "TotalConditions": len(r.conditions_met) + len(r.conditions_failed),
            })

        df = pd.DataFrame(rows)
        csv_str = df.to_csv(index=False, encoding="utf-8-sig")

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8-sig") as f:
                f.write(csv_str)

        return csv_str


# ═══════════════════════════════════════════════════════════════
# 3. 排程整合 (Cron Job 入口)
# ═══════════════════════════════════════════════════════════════

def run_daily_scan(
    tickers: list[str] | None = None,
    *,
    output: str | None = None,
    fmt: str = "text",
    progress: bool = True,
) -> str:
    """執行每日掃描並回傳報表 (Cron Job 可使用此函式).

    Args:
        tickers: 掃描清單
        output: 輸出檔案路徑 (None=僅回傳字串)
        fmt: "text", "html", "csv"
        progress: 顯示進度

    Returns:
        報表文字/HTML/CSV
    """
    scanner = DailyScanner()
    report = scanner.scan(tickers, progress=progress)

    if fmt == "html":
        result = ReportGenerator.html_report(report, output_path=output)
    elif fmt == "csv":
        result = ReportGenerator.csv_report(report, output_path=output)
    else:
        result = ReportGenerator.text_report(report)
        if output:
            with open(output, "w", encoding="utf-8") as f:
                f.write(result)

    return result


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TWSE 每日量化掃描")
    parser.add_argument("--watchlist", help="掃描清單 (逗號分隔)")
    parser.add_argument("--all", action="store_true", help="掃描全市場")
    parser.add_argument("--report", nargs="?", const="auto", help="輸出報表路徑")
    parser.add_argument("--format", choices=["text", "html", "csv"],
                        default="text", help="報表格式")
    parser.add_argument("--cron", action="store_true",
                        help="Cron 模式 (靜默執行，輸出 HTML 到 reports/)")
    args = parser.parse_args()

    # 決定掃描清單
    tickers = None
    if args.watchlist:
        tickers = [t.strip() for t in args.watchlist.split(",")]
    elif args.all:
        tickers = None  # Full market scan (needs ticker list)

    # Cron 模式
    if args.cron:
        today = date.today().isoformat()
        html_path = os.path.join(REPORT_DIR, f"scan_{today}.html")
        csv_path = os.path.join(REPORT_DIR, f"scan_{today}.csv")

        print(f"[Cron] TWSE 掃描開始 {today}")
        t0 = time.time()

        run_daily_scan(tickers, output=html_path, fmt="html", progress=False)
        run_daily_scan(tickers, output=csv_path, fmt="csv", progress=False)

        elapsed = time.time() - t0
        print(f"[Cron] 完成 ({elapsed:.0f}秒)")
        print(f"  HTML: {html_path}")
        print(f"  CSV:  {csv_path}")
        sys.exit(0)

    # 一般模式
    output_path = None
    if args.report:
        if args.report == "auto":
            ext = {"text": "txt", "html": "html", "csv": "csv"}[args.format]
            output_path = os.path.join(
                REPORT_DIR, f"scan_{date.today().isoformat()}.{ext}",
            )
        else:
            output_path = args.report

    result = run_daily_scan(tickers, output=output_path, fmt=args.format)
    print(result)
