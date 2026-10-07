"""
ファンダメンタルズ判定モジュール
==================================================
検証ノートでテクニカル的に合格した銘柄に対して、
「財務的に地雷ではないか」を判定します。

設計方針
--------
日本株の財務データはAPIごとに欠損の仕方が違うため、
データの取得（Provider）と判定（Filter）を分離しています。
将来 J-Quants API やローカルCSVに差し替えるときは、
FinancialDataProvider を継承したクラスを1つ書いて渡すだけで済みます。

【重要】この判定は検証できません。
過去の各時点の財務データが取得できないため、
「この基準で除外すると成績が良くなる」ことを数字で示せません。
テクニカルのルールとは信頼度の質が違うものとして扱ってください。
用途は「買ってはいけない銘柄を外す」拒否権に限定してください。

使い方
------
    pip install yfinance pandas
    python fundamental_filter.py

結果は画面に出るほか、fundamentals.json に書き出されます。
それを jirai-check.html に読み込ませると一覧で見られます。
"""

from __future__ import annotations

import json
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from typing import List, Optional, Dict, Any

try:
    import yfinance as yf
    import pandas as pd
except ImportError:
    print("必要なライブラリが入っていません:  pip install yfinance pandas")
    sys.exit(1)


# ============================================================
# 調べたい銘柄
# ============================================================
TICKERS: Dict[str, str] = {
    "7532": "パンパシHD",
    "1911": "住友林業",
    "6871": "日本マイクロニクス",
    "4588": "オンコリス",
    "4007": "レゾナック",
    "2607": "不二製油グループ本社",
    "3932": "アカツキ",
    "4519": "中外製薬",
    "4565": "そーせい",
    "6703": "沖電気工業",
    "8051": "山善",
    "9031": "西日本鉄道",
    "9519": "レノバ",


}

HOLDING_DAYS = 12      # 保有予定日数（暦日）。この間に決算があるかを見る


# ============================================================
# 取得したデータの入れ物
# ============================================================
@dataclass
class Financials:
    """プロバイダが返す生データ。数値はすべて新しい順に並べる"""
    code: str
    name: str = ""
    operating_income_annual: List[float] = field(default_factory=list)
    operating_income_quarterly: List[float] = field(default_factory=list)
    revenue_annual: List[float] = field(default_factory=list)
    revenue_quarterly: List[float] = field(default_factory=list)
    net_income_annual: List[float] = field(default_factory=list)
    total_assets: Optional[float] = None
    total_equity: Optional[float] = None
    market_cap: Optional[float] = None
    per: Optional[float] = None
    pbr: Optional[float] = None
    next_earnings: Optional[str] = None        # "YYYY-MM-DD"
    ex_dividend: Optional[str] = None          # 配当の権利落ち日
    atr_ratio: Optional[float] = None          # 直近ATR ÷ 過去1年の平均ATR
    ret_1m: Optional[float] = None             # 直近1ヶ月の騰落率（％）
    limit_days: Optional[int] = None           # 直近60日でストップ幅級に動いた日数
    news: List[Dict[str, str]] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)   # 取れなかった項目


# ============================================================
# プロバイダの抽象基底クラス
# ============================================================
class FinancialDataProvider(ABC):
    """財務データの取得元。差し替えるときはこれを継承する"""

    @abstractmethod
    def fetch(self, code: str, name: str = "") -> Financials:
        """1銘柄ぶんまとめて取得する"""

    # --- 以下は fetch の結果から導く。共通実装で足りるので継承先で書き直す必要はない ---

    def get_operating_income(self, fin: Financials) -> Optional[float]:
        """最新の営業利益。四半期があればそちらを優先する"""
        if fin.operating_income_quarterly:
            return fin.operating_income_quarterly[0]
        if fin.operating_income_annual:
            return fin.operating_income_annual[0]
        return None

    def get_equity_ratio(self, fin: Financials) -> Optional[float]:
        """自己資本比率（％）"""
        if fin.total_assets and fin.total_equity and fin.total_assets != 0:
            return fin.total_equity / fin.total_assets * 100
        return None

    def get_revenue_yoy(self, fin: Financials) -> Optional[float]:
        """売上の前年同期比（％）。四半期が5期ぶん無ければ通期で代用する"""
        q = fin.revenue_quarterly
        if len(q) >= 5 and q[4]:
            return (q[0] - q[4]) / abs(q[4]) * 100
        a = fin.revenue_annual
        if len(a) >= 2 and a[1]:
            return (a[0] - a[1]) / abs(a[1]) * 100
        return None

    def get_days_to_exdiv(self, fin: Financials) -> Optional[int]:
        """次の権利落ち日までの日数。過去の日付しか無い場合は不明として扱う"""
        if not fin.ex_dividend:
            return None
        try:
            d = datetime.strptime(fin.ex_dividend[:10], "%Y-%m-%d").date()
        except ValueError:
            return None
        days = (d - date.today()).days
        return None if days < 0 else days

    def get_days_to_earnings(self, fin: Financials) -> Optional[int]:
        """次の決算発表までの日数"""
        if not fin.next_earnings:
            return None
        try:
            d = datetime.strptime(fin.next_earnings, "%Y-%m-%d").date()
        except ValueError:
            return None
        return (d - date.today()).days


# ============================================================
# yfinance 実装
# ============================================================
class YFinanceProvider(FinancialDataProvider):
    """yfinance から取得する。日本株は欠損が多いので missing に記録する"""

    SUFFIX = ".T"

    @staticmethod
    def _row(df, *names) -> List[float]:
        """決算データの行名はバージョンで変わるため候補を順に探す"""
        if df is None or getattr(df, "empty", True):
            return []
        for want in names:
            for idx in df.index:
                if str(idx).strip().lower() == want.lower():
                    vals = [float(v) for v in df.loc[idx].tolist() if pd.notna(v)]
                    if vals:
                        return vals
        return []

    def _earnings_date(self, t) -> Optional[str]:
        # calendar から
        try:
            cal = t.calendar
            if isinstance(cal, dict):
                ds = cal.get("Earnings Date") or []
                if ds:
                    d = ds[0]
                    return d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10]
            elif cal is not None and not getattr(cal, "empty", True):
                v = cal.iloc[0, 0]
                return str(v)[:10]
        except Exception:
            pass
        # get_earnings_dates から未来の最も近いもの
        try:
            ed = t.get_earnings_dates(limit=12)
            if ed is not None and not ed.empty:
                today = pd.Timestamp.now(tz=ed.index.tz)
                future = [i for i in ed.index if i >= today]
                if future:
                    return min(future).strftime("%Y-%m-%d")
        except Exception:
            pass
        return None

    def fetch(self, code: str, name: str = "") -> Financials:
        t = yf.Ticker(f"{code}{self.SUFFIX}")
        fin = Financials(code=code, name=name)

        try:
            info = t.info or {}
        except Exception:
            info = {}
            fin.missing.append("info")

        fin.market_cap = info.get("marketCap")
        fin.per = info.get("trailingPE")
        fin.pbr = info.get("priceToBook")

        try:
            ann = t.financials
        except Exception:
            ann = None
            fin.missing.append("通期決算")
        try:
            qtr = t.quarterly_financials
        except Exception:
            qtr = None
            fin.missing.append("四半期決算")
        try:
            bs = t.balance_sheet
        except Exception:
            bs = None
            fin.missing.append("貸借対照表")

        fin.operating_income_annual = self._row(ann, "Operating Income", "EBIT")
        fin.operating_income_quarterly = self._row(qtr, "Operating Income", "EBIT")
        fin.revenue_annual = self._row(ann, "Total Revenue", "Operating Revenue")
        fin.revenue_quarterly = self._row(qtr, "Total Revenue", "Operating Revenue")
        fin.net_income_annual = self._row(ann, "Net Income", "Net Income Common Stockholders")

        assets = self._row(bs, "Total Assets")
        equity = self._row(bs, "Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity")
        fin.total_assets = assets[0] if assets else None
        fin.total_equity = equity[0] if equity else None

        if not fin.operating_income_quarterly and not fin.operating_income_annual:
            fin.missing.append("営業利益")
        if fin.total_assets is None or fin.total_equity is None:
            fin.missing.append("自己資本比率の材料")

        fin.next_earnings = self._earnings_date(t)
        if not fin.next_earnings:
            fin.missing.append("決算予定日")

        # 権利落ち日。info に無ければ過去の配当履歴から推定する
        ed = info.get("exDividendDate")
        if ed:
            try:
                fin.ex_dividend = (datetime.fromtimestamp(ed) if isinstance(ed, (int, float))
                                   else pd.to_datetime(ed)).strftime("%Y-%m-%d")
            except Exception:
                pass
        if not fin.ex_dividend:
            try:
                div = t.dividends
                if div is not None and len(div) >= 2:
                    last = div.index[-1]
                    # 直近2回の間隔を足して次回を見積もる
                    gap = (div.index[-1] - div.index[-2]).days
                    nxt = last + timedelta(days=gap)
                    while nxt.date() < date.today():
                        nxt = nxt + timedelta(days=gap)
                    fin.ex_dividend = nxt.strftime("%Y-%m-%d") + "（推定）"
            except Exception:
                pass

        # 値動きが今どれだけ荒れているか
        try:
            px = t.history(period="1y", interval="1d", auto_adjust=False)
            if px is not None and not px.empty:
                px = px.dropna(subset=["High", "Low", "Close"])
            if px is not None and len(px) > 60:
                h, l, c = px["High"], px["Low"], px["Close"]
                prev = c.shift(1)
                tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
                atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
                recent, avg = float(atr.iloc[-1]), float(atr.mean())
                fin.atr_ratio = recent / avg if avg > 0 else None
                if len(c) >= 21:
                    fin.ret_1m = (float(c.iloc[-1]) / float(c.iloc[-21]) - 1) * 100
                # 1日で15%以上動いた日＝制限値幅に近い動き
                mv = (c / c.shift(1) - 1).abs().tail(60)
                fin.limit_days = int((mv >= 0.15).sum())
        except Exception:
            fin.missing.append("値動きの履歴")

        try:
            for n in (t.news or [])[:5]:
                c = n.get("content", n)
                ts = c.get("pubDate") or n.get("providerPublishTime")
                if isinstance(ts, (int, float)):
                    ts = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                fin.news.append({
                    "title": str(c.get("title", ""))[:120],
                    "publisher": str((c.get("provider") or {}).get("displayName")
                                     or n.get("publisher") or "")[:40],
                    "date": str(ts)[:10] if ts else "",
                    "link": str((c.get("canonicalUrl") or {}).get("url") or n.get("link") or ""),
                })
        except Exception:
            pass

        return fin


# ============================================================
# 判定
# ============================================================
@dataclass
class FilterConfig:
    min_equity_ratio: float = 20.0     # 自己資本比率の下限（％）
    holding_days: int = HOLDING_DAYS   # 保有予定日数
    require_revenue_growth: bool = False   # 売上成長を必須にするか（既定は加点のみ）
    max_atr_ratio: float = 2.0         # 直近ATRが平常の何倍までなら許すか
    min_ret_1m: float = -30.0          # 直近1ヶ月の下落率の下限（％）
    max_limit_days: int = 3            # 直近60日で15%以上動いた日数の上限


@dataclass
class Verdict:
    code: str
    name: str
    passed: bool                       # 致命傷なし
    score: int                         # 加点（0〜1）
    fails: List[str] = field(default_factory=list)
    warns: List[str] = field(default_factory=list)
    pluses: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    news: List[Dict[str, str]] = field(default_factory=list)

    @property
    def label(self) -> str:
        if self.fails:
            return "地雷"
        if self.warns:
            return "要注意"
        return "問題なし"


class FundamentalFilter:
    """プロバイダを注入して使う。判定ロジックはここだけに置く"""

    def __init__(self, provider: FinancialDataProvider, config: Optional[FilterConfig] = None):
        self.provider = provider
        self.cfg = config or FilterConfig()

    def evaluate(self, code: str, name: str = "") -> Verdict:
        p, cfg = self.provider, self.cfg
        fin = p.fetch(code, name)

        op = p.get_operating_income(fin)
        eq = p.get_equity_ratio(fin)
        yoy = p.get_revenue_yoy(fin)
        d2e = p.get_days_to_earnings(fin)
        d2x = p.get_days_to_exdiv(fin)

        fails, warns, pluses = [], [], []

        # --- ネガティブフィルター：ひとつでも当たれば地雷 ---
        if op is None:
            warns.append("営業利益が取得できない")
        elif op < 0:
            fails.append("本業が赤字（営業赤字）")

        if eq is None:
            warns.append("自己資本比率が取得できない")
        elif eq < cfg.min_equity_ratio:
            fails.append(f"自己資本比率 {eq:.0f}%（{cfg.min_equity_ratio:.0f}%未満）")

        # --- 決算跨ぎ：逆指値では防げないので独立して扱う ---
        if d2e is None:
            warns.append("決算予定日が不明")
        elif 0 <= d2e <= cfg.holding_days:
            fails.append(f"保有期間内に決算（あと{d2e}日）")
        elif d2e <= cfg.holding_days + 7:
            warns.append(f"決算が近い（あと{d2e}日）")

        # --- 権利落ち：保有期間内なら配当ぶん機械的に下がる ---
        if d2x is not None and 0 <= d2x <= cfg.holding_days:
            warns.append(f"保有期間内に権利落ち（あと{d2x}日）")

        # --- 値動きが荒れすぎていないか ---
        if fin.atr_ratio is None:
            warns.append("値動きの履歴が取得できない")
        elif fin.atr_ratio > cfg.max_atr_ratio:
            fails.append(f"値動きが平常の{fin.atr_ratio:.1f}倍に荒れている")
        if fin.limit_days is not None and fin.limit_days > cfg.max_limit_days:
            fails.append(f"直近60日で15%以上動いた日が{fin.limit_days}回")
        if fin.ret_1m is not None and fin.ret_1m < cfg.min_ret_1m:
            fails.append(f"1ヶ月で{fin.ret_1m:.0f}%の下落（崩落の途中）")

        # --- ポジティブフィルター：加点のみ ---
        if yoy is None:
            warns.append("売上成長率が取得できない")
        elif yoy > 0:
            pluses.append(f"増収 +{yoy:.1f}%")
        else:
            if cfg.require_revenue_growth:
                fails.append(f"減収 {yoy:.1f}%")
            else:
                warns.append(f"減収 {yoy:.1f}%")

        # 参考情報（判定には使わない）
        if fin.net_income_annual and fin.net_income_annual[0] < 0 and op is not None and op >= 0:
            warns.append("営業黒字だが最終赤字（特別損失など）")

        return Verdict(
            code=code, name=name or fin.name,
            passed=not fails,
            score=len(pluses),
            fails=fails, warns=warns, pluses=pluses,
            metrics={
                "営業利益": op, "自己資本比率": eq, "売上YoY": yoy,
                "決算まで日数": d2e, "決算予定日": fin.next_earnings,
                "権利落ちまで日数": d2x, "権利落ち日": fin.ex_dividend,
                "値動きの倍率": fin.atr_ratio, "1ヶ月騰落": fin.ret_1m,
                "急変した日数": fin.limit_days,
                "時価総額": fin.market_cap, "PER": fin.per, "PBR": fin.pbr,
                "売上推移": fin.revenue_annual[:4],
                "営業利益推移": fin.operating_income_annual[:4],
                "純利益推移": fin.net_income_annual[:4],
                "取得できなかった項目": fin.missing,
            },
            news=fin.news,
        )


# ============================================================
# 表示
# ============================================================
def money(v) -> str:
    if v is None:
        return "—"
    a = abs(v)
    if a >= 1e12:
        return f"{v/1e12:.2f}兆"
    if a >= 1e8:
        return f"{v/1e8:,.0f}億"
    return f"{v:,.0f}"


def main():
    provider = YFinanceProvider()              # ここを差し替えれば取得元が変わる
    filt = FundamentalFilter(provider, FilterConfig(holding_days=HOLDING_DAYS))

    print(f"ファンダメンタルズ判定  {len(TICKERS)} 銘柄  （保有想定 {HOLDING_DAYS} 日）\n")
    results: List[Verdict] = []
    for code, name in TICKERS.items():
        print(f"  {code} {name} ...", end=" ", flush=True)
        try:
            v = filt.evaluate(code, name)
            print(v.label)
            results.append(v)
        except Exception as e:
            print(f"失敗（{type(e).__name__}: {e}）")
        time.sleep(0.4)

    if not results:
        print("\n1件も取得できませんでした。")
        return

    order = {"地雷": 0, "要注意": 1, "問題なし": 2}
    results.sort(key=lambda v: (order[v.label], -v.score))

    print("\n" + "=" * 86)
    print(f"{'コード':<7}{'銘柄':<14}{'営業利益':>10}{'自己資本':>8}{'決算':>7}{'権利落':>7}{'荒れ':>6}{'1ヶ月':>8}  判定")
    print("-" * 100)
    for v in results:
        m = v.metrics
        op = money(m["営業利益"])
        eq = f"{m['自己資本比率']:.0f}%" if m["自己資本比率"] is not None else "—"
        de = f"{m['決算まで日数']}日" if m["決算まで日数"] is not None else "不明"
        dx = f"{m['権利落ちまで日数']}日" if m["権利落ちまで日数"] is not None else "—"
        ar = f"{m['値動きの倍率']:.1f}倍" if m["値動きの倍率"] is not None else "—"
        r1 = f"{m['1ヶ月騰落']:+.0f}%" if m["1ヶ月騰落"] is not None else "—"
        note = "・".join(v.fails or v.warns or v.pluses) or "—"
        print(f"{v.code:<7}{v.name:<14}{op:>10}{eq:>8}{de:>7}{dx:>7}{ar:>6}{r1:>8}  {v.label}：{note}")

    ng = [v for v in results if v.fails]
    if ng:
        print("\n" + "=" * 86)
        print("地雷と判定された銘柄の詳細（新しい順）")
        print("=" * 86)
        for v in ng:
            print(f"\n{v.code} {v.name}  — {'・'.join(v.fails)}")
            for label, key in [("売上", "売上推移"), ("営業利益", "営業利益推移"), ("純利益", "純利益推移")]:
                vals = v.metrics.get(key) or []
                if vals:
                    print(f"  {label:<6}" + "  ".join(money(x) for x in vals))

    ok = [v for v in results if v.passed]
    print("\n" + "-" * 86)
    print(f"問題なし {sum(1 for v in results if v.label=='問題なし')} / "
          f"要注意 {sum(1 for v in results if v.label=='要注意')} / "
          f"地雷 {len(ng)}")
    if ok:
        print("通過した銘柄: " + "、".join(f"{v.code} {v.name}" for v in ok))

    # --- Webページ用に書き出す ---
    def clean(v):
        """NaN や inf は JSON として無効なので None に変換する"""
        if isinstance(v, float):
            return None if (v != v or v in (float("inf"), float("-inf"))) else v
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        return v

    out = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "holding_days": HOLDING_DAYS,
        "results": [
            clean({**asdict(v), "label": v.label}) for v in results
        ],
    }
    with open("fundamentals.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str, allow_nan=False)
    print("\nfundamentals.json に書き出しました。jirai-check.html に読み込ませてください。")

    print("\nこの判定は検証されていません。買ってよい銘柄を選ぶものではなく、")
    print("買ってはいけない銘柄を外すためだけに使ってください。")


if __name__ == "__main__":
    main()
