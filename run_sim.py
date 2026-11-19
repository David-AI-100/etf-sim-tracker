from datetime import datetime, timezone, timedelta
import os
import numpy as np
import pandas as pd
import yfinance as yf

INITIAL_CASH = 100000.0
LOG_CSV_PATH = "data/etf_sim_log.csv"
SUMMARY_MD_PATH = "summary.md"

ETF_LIST = [
    {
        "name": "KODEX 200 위클리 커버드콜",
        "code": "498400",
        "benchmark_type": "KS200",
    },
    {
        "name": "TIGER 미국배당 다우존스타겟데일리커버드콜",
        "code": "0008S0",
        "benchmark_type": "SP500",
    },
    {
        "name": "KODEX 삼성전자SK하이닉스 채권혼합50",
        "code": "0177N0",
        "benchmark_type": "KS200",
    },
    {
        "name": "TIGER 미국우주테크",
        "code": "0183J0",
        "benchmark_type": "SP500",
    },
]

COLUMNS = [
    "etf_code",
    "etf_name",
    "trade_date",
    "close",
    "rsi",
    "signal",
    "action",
    "qty_delta",
    "cash",
    "shares",
    "avg_price",
    "total_asset",
    "return_pct",
]


def fetch_yf_series(ticker: str, start: str = "2023-01-01") -> pd.DataFrame:
    df = yf.download(ticker, start=start, auto_adjust=False, progress=False, threads=False)
    if df is None or len(df) == 0:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    df = df.dropna(subset=["Close"]).copy()
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df


def load_kr_etf_yf(code: str, start: str = "2023-01-01") -> pd.DataFrame:
    candidates = [code] if code.endswith((".KS", ".KQ")) else [f"{code}.KS", f"{code}.KQ"]
    for sym in candidates:
        df = fetch_yf_series(sym, start=start)
        if not df.empty:
            return df
    return pd.DataFrame()


def make_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = df["Close"].diff(1).fillna(0)
    up = delta.where(delta > 0, 0.0)
    down = -delta.where(delta < 0, 0.0)
    au = up.ewm(com=period - 1, min_periods=1, adjust=False).mean()
    ad = down.ewm(com=period - 1, min_periods=1, adjust=False).mean()
    rs = au / ad.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def make_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev_factor: float = 2.0) -> pd.DataFrame:
    out = df.copy()
    out["MA20"] = out["Close"].rolling(window=period, min_periods=1).mean()
    std_dev = out["Close"].rolling(window=period, min_periods=1).std().fillna(0)
    out["Upper"] = out["MA20"] + (std_dev * std_dev_factor)
    out["Lower"] = out["MA20"] - (std_dev * std_dev_factor)
    return out


def calc_signal_hit_rate(df_signal: pd.DataFrame, horizon: int = 5):
    buy_indices = np.where(df_signal["Signal"].values == 1)[0]
    closes = df_signal["Close"].values
    evaluated = 0
    hits = 0
    for idx in buy_indices:
        future_idx = idx + horizon
        if future_idx < len(closes):
            evaluated += 1
            if closes[future_idx] > closes[idx]:
                hits += 1
    if evaluated == 0:
        return None, 0, 0
    return (hits / evaluated) * 100.0, hits, evaluated


def load_log(csv_path: str) -> pd.DataFrame:
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            for col in COLUMNS:
                if col not in df.columns:
                    df[col] = np.nan
            return df[COLUMNS].copy()
        except Exception:
            pass
    return pd.DataFrame(columns=COLUMNS)


def save_log(log_df: pd.DataFrame, csv_path: str):
    parent = os.path.dirname(os.path.abspath(csv_path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    log_df.to_csv(csv_path, index=False, encoding="utf-8-sig")


def step_simulation(log_df: pd.DataFrame, etf_code: str, etf_name: str, trade_date_str: str, close: float, rsi: float, signal: int):
    sub = log_df[log_df["etf_code"].astype(str) == str(etf_code)]
    if not sub.empty and trade_date_str in set(sub["trade_date"].astype(str)):
        prior = sub[sub["trade_date"].astype(str) < trade_date_str]
        if prior.empty:
            cash, shares, avg_price = INITIAL_CASH, 0, 0.0
        else:
            p_last = prior.iloc[-1]
            cash, shares, avg_price = float(p_last["cash"]), int(p_last["shares"]), float(p_last["avg_price"])
    elif not sub.empty:
        last = sub.iloc[-1]
        cash, shares, avg_price = float(last["cash"]), int(last["shares"]), float(last["avg_price"])
    else:
        cash, shares, avg_price = INITIAL_CASH, 0, 0.0

    today_buy = 0
    today_sell = 0
    action = "관망(HOLD)"

    if signal == 1 and cash >= close:
        buy_qty = int((INITIAL_CASH * 0.3) // close)
        if buy_qty < 1 and cash >= close:
            buy_qty = 1
        cost = buy_qty * close
        if buy_qty > 0 and cash >= cost:
            total_cost = (shares * avg_price) + cost
            shares += buy_qty
            avg_price = total_cost / shares if shares > 0 else 0.0
            cash -= cost
            today_buy = buy_qty
            action = f"매수(+{buy_qty}주)"
    elif signal == -1 and shares > 0:
        sell_qty = shares // 2
        if sell_qty < 1:
            sell_qty = shares
        cash += sell_qty * close
        shares -= sell_qty
        today_sell = sell_qty
        action = f"매도(-{sell_qty}주)"
        if shares == 0:
            avg_price = 0.0

    total_asset = cash + (shares * close)
    return_pct = ((total_asset - INITIAL_CASH) / INITIAL_CASH) * 100.0

    row = {
        "etf_code": str(etf_code),
        "etf_name": etf_name,
        "trade_date": trade_date_str,
        "close": round(float(close), 2),
        "rsi": round(float(rsi), 2),
        "signal": int(signal),
        "action": action,
        "qty_delta": int(today_buy - today_sell),
        "cash": round(float(cash), 0),
        "shares": int(shares),
        "avg_price": round(float(avg_price), 2),
        "total_asset": round(float(total_asset), 0),
        "return_pct": round(float(return_pct), 2),
    }

    mask = (log_df["etf_code"].astype(str) == str(etf_code)) & (log_df["trade_date"].astype(str) == trade_date_str)
    updated_df = pd.concat([log_df[~mask], pd.DataFrame([row])], ignore_index=True)
    updated_df = updated_df.sort_values(["etf_code", "trade_date"]).reset_index(drop=True)
    return updated_df, row


def main():
    log_df = load_log(LOG_CSV_PATH)
    kst_now = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M KST")
    md_lines = [
        "# ETF 10만 원 시뮬레이션 누적 요약",
        f"- 최종 갱신: {kst_now}",
        "",
        "| 종목명 | 코드 | 기준일 | 종가 | RSI | 현금 잔고 | 보유 수량 | 평가자산 | 수익률 | 5일 적중률 | 오늘 판정 |",
        "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- |",
    ]

    for etf in ETF_LIST:
        df_etf = load_kr_etf_yf(etf["code"], start="2023-01-01")
        if df_etf.empty:
            md_lines.append(f"| {etf['name']} | {etf['code']} | - | - | - | - | - | - | - | 데이터 없음 | 조회 실패 |")
            continue

        df_etf["RSI"] = make_rsi(df_etf)
        df_etf = make_bollinger_bands(df_etf)
        df_etf["Signal"] = 0
        buy_cond = (df_etf["RSI"] <= 35) & (df_etf["Close"] <= df_etf["Lower"] * 1.01)
        sell_cond = (df_etf["RSI"] >= 65) | (df_etf["Close"] >= df_etf["Upper"] * 0.99)
        df_etf.loc[buy_cond, "Signal"] = 1
        df_etf.loc[sell_cond, "Signal"] = -1

        hit_rate, hits, evaluated = calc_signal_hit_rate(df_etf, horizon=5)
        hit_str = f"{hit_rate:.1f}% ({hits}/{evaluated})" if hit_rate is not None else "집계 중"

        latest_date = df_etf.index[-1].strftime("%Y-%m-%d")
        close = float(df_etf["Close"].iloc[-1])
        rsi = float(df_etf["RSI"].iloc[-1])
        signal = int(df_etf["Signal"].iloc[-1])

        log_df, row = step_simulation(log_df, etf["code"], etf["name"], latest_date, close, rsi, signal)
        sign = "+" if row["return_pct"] > 0 else ""

        md_lines.append(
            f"| {etf['name']} | {etf['code']} | {latest_date} | {close:,.0f}원 | {rsi:.1f} | "
            f"{row['cash']:,.0f}원 | {row['shares']:,}주 | {row['total_asset']:,.0f}원 | "
            f"{sign}{row['return_pct']:.2f}% | {hit_str} | {row['action']} |"
        )

    md_lines.extend([
        "",
        "※ 본 정보와 시뮬레이션은 자산관리 및 투자 판단을 돕기 위한 참고 자료이며, 최종 투자 결정과 책임은 투자자 본인에게 있습니다.",
    ])

    save_log(log_df, LOG_CSV_PATH)
    with open(SUMMARY_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")


if __name__ == "__main__":
    main()
