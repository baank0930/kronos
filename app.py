"""Kronos mini/small research application for Streamlit.

Uses a namespaced, locally vendored snapshot of the upstream model implementation.
Not an order-routing or investment-advice application.
"""
from __future__ import annotations

import re
import sys
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import streamlit as st
import torch
import yfinance as yf

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from kronos_engine import Kronos, KronosTokenizer, KronosPredictor

MODEL_SPEC = {
    "mini": {
        "tokenizer": "NeoQuasar/Kronos-Tokenizer-2k",
        "tokenizer_revision": "26966d0035065a0cae0ebad7af8ece35bc1fb51c",
        "model": "NeoQuasar/Kronos-mini",
        "model_revision": "f4e68697d9d5aed55cef5c96aabc3376bcad9f81",
    },
    "small": {
        "tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
        "tokenizer_revision": "0e0117387f39004a9016484a186a908917e22426",
        "model": "NeoQuasar/Kronos-small",
        "model_revision": "901c26c1332695a2a8f243eb2f37243a37bea320",
    },
}
LOOKBACK = 64
SEED = 42
PATH_COUNT = 5
THRESHOLD_PCT = 0.5


# 以下はYahoo Financeでのシンボル例。商品先物には限月・ロール等の注意がある。
FUTURES = {
    "ES=F": ("CME_Equity", "points", "S&P 500先物"),
    "NQ=F": ("CME_Equity", "points", "Nasdaq 100先物"),
    "YM=F": ("CME_Equity", "points", "ダウ先物"),
    "RTY=F": ("CME_Equity", "points", "ラッセル2000先物"),
    "NIY=F": ("CME_Equity", "points", "CME日経225先物・円建て"),
    "NKD=F": ("CME_Equity", "points", "CME日経225先物・ドル建て"),
    "GC=F": ("CMEGlobex_Gold", "USD/oz", "金先物"),
    "SI=F": ("CMEGlobex_Silver", "USD/oz", "銀先物"),
    "CL=F": ("CMEGlobex_Crude", "USD/bbl", "WTI原油先物"),
    "NG=F": ("CMEGlobex_NatGas", "USD/MMBtu", "天然ガス先物"),
}
# N225はCME円建て日経225先物。^N225は日経225指数で別物。
ALIASES = {"N225": "NIY=F", "USDJPY": "JPY=X", "USD/JPY": "JPY=X"}


def market_for(ticker: str) -> dict:
    if ticker in FUTURES:
        calendar, unit, label = FUTURES[ticker]
        return {"calendar": calendar, "tz": "America/New_York", "currency": unit,
                "label": label, "kind": "futures"}
    if ticker == "JPY=X":
        return {"calendar": "24/5", "tz": "Europe/London", "currency": "JPY per USD",
                "label": "米ドル／円・現物為替", "kind": "forex"}
    if ticker == "^GSPC":
        return {"calendar": "NYSE", "tz": "America/New_York", "currency": "points",
                "label": "S&P 500指数", "kind": "index"}
    if ticker == "^N225":
        return {"calendar": "JPX", "tz": "Asia/Tokyo", "currency": "points",
                "label": "日経225指数", "kind": "index"}
    if re.fullmatch(r"[A-Z0-9]{4,6}\.T", ticker):
        return {"calendar": "JPX", "tz": "Asia/Tokyo", "currency": "JPY", "label": "日本"}
    if re.fullmatch(r"[A-Z0-9]{4,6}\.(TW|TWO)", ticker):
        return {"calendar": "XTAI", "tz": "Asia/Taipei", "currency": "TWD", "label": "台湾"}
    if re.fullmatch(r"[A-Z][A-Z0-9]{0,9}(?:-[A-Z])?", ticker):
        return {"calendar": "NYSE", "tz": "America/New_York", "currency": "USD", "label": "米国"}
    raise ValueError("未対応の形式です。例：SPY、1320.T、0050.TW、ES=F、NIY=F、JPY=X。")


@st.cache_resource(show_spinner=False)
def model_for(name: str):
    spec = MODEL_SPEC[name]
    tokenizer = KronosTokenizer.from_pretrained(spec["tokenizer"], revision=spec["tokenizer_revision"])
    model = Kronos.from_pretrained(spec["model"], revision=spec["model_revision"])
    # Streamlit Community CloudではCPU前提。再読み込みせず全セッションで共有。
    return KronosPredictor(model, tokenizer, device="cpu", max_context=512)


@st.cache_resource(show_spinner=False)
def inference_lock():
    # torchの乱数が共有されるため、推論のseed設定と実行を他セッションと直列化する。
    return threading.RLock()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_prices(ticker: str) -> pd.DataFrame:
    raw = yf.download(ticker, period="3y", interval="1d", auto_adjust=True,
                      progress=False, threads=False, multi_level_index=False, timeout=25)
    if raw is None or raw.empty:
        raise ValueError("株価データを取得できません。コードまたはYahoo側の通信制限を確認してください。")
    if isinstance(raw.columns, pd.MultiIndex):
        raise ValueError("yfinanceの列形式が想定と異なります。yfinanceのバージョンを確認してください。")
    raw.columns = [str(c).lower() for c in raw.columns]
    needed = ["open", "high", "low", "close", "volume"]
    if not set(needed).issubset(raw.columns):
        raise ValueError(f"必要な列が不足しています: {list(raw.columns)}")
    return raw[needed].copy()


def prepare_data(ticker: str, horizon: int) -> dict:
    setting = market_for(ticker)
    raw = fetch_prices(ticker).copy()
    idx = pd.DatetimeIndex(raw.index)
    if idx.tz is not None:
        idx = idx.tz_convert(setting["tz"]).tz_localize(None)
    raw.index = idx.normalize()
    raw = raw.loc[~raw.index.duplicated(keep="last")].sort_index()
    raw = raw.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    raw = raw[(raw[["open", "high", "low", "close"]] > 0).all(axis=1) & (raw["volume"] >= 0)]
    cal = mcal.get_calendar(setting["calendar"])
    now = pd.Timestamp.now(tz="UTC")
    today = now.tz_convert(setting["tz"]).date()
    recent = cal.schedule(start_date=str(today - timedelta(days=45)), end_date=str(today))
    closed = recent[recent["market_close"] <= now]
    if closed.empty:
        raise ValueError("直近の取引終了日を取引カレンダーで確認できません。")
    last_closed = pd.Timestamp(closed.index[-1])
    if last_closed.tzinfo is not None:
        last_closed = last_closed.tz_localize(None)
    last_closed = last_closed.normalize()
    raw = raw.loc[raw.index <= last_closed]
    if len(raw) < LOOKBACK:
        raise ValueError(f"完了済みの日足は{len(raw)}本です。最低{LOOKBACK}本必要です。")
    last = pd.Timestamp(raw.index[-1]).normalize()
    future = cal.schedule(start_date=str((last + pd.Timedelta(days=1)).date()),
                          end_date=str((last + pd.Timedelta(days=120)).date()))
    if len(future) < horizon:
        raise ValueError("予測先の取引日を十分取得できません。")
    dates = []
    for value in future.index[:horizon]:
        item = pd.Timestamp(value)
        dates.append((item.tz_localize(None) if item.tzinfo is not None else item).normalize())
    hist = raw.tail(LOOKBACK)
    zero_volume = int((hist["volume"] == 0).sum())
    flat_ohlc = int((hist[["open", "high", "low", "close"]].nunique(axis=1) == 1).sum())
    return {"raw": raw, "hist": hist, "setting": setting, "last": last,
            "last_closed": last_closed, "future_dates": dates, "local_today": today,
            "zero_volume": zero_volume, "flat_ohlc": flat_ohlc}


def set_prediction_origin(prepared: dict, mode: int, specified_date: date, requested_horizon: int) -> dict:
    """Cut off the model's data at the origin; keep later rows only for display/evaluation."""
    raw_full = prepared["raw"]
    latest_available = pd.Timestamp(raw_full.index[-1])
    if mode == -1:
        if specified_date > prepared["local_today"]:
            raise ValueError("起点日には未来の日付を指定できません。")
        matches = np.flatnonzero(raw_full.index <= pd.Timestamp(specified_date))
        if not len(matches):
            raise ValueError("指定日以前のデータがありません。日付を見直してください。")
        i = int(matches[-1])
    else:
        i = len(raw_full) - 1 - int(mode)
    if i < LOOKBACK - 1:
        raise ValueError(f"指定した起点までに{LOOKBACK}本の有効な日足がありません。")
    anchor = pd.Timestamp(raw_full.index[i]).normalize()
    holdout = raw_full.iloc[i + 1:].copy()
    input_raw = raw_full.iloc[:i + 1].copy()
    # 既に取得済みの1〜5本を比較したい場合は、自動的に予測長を延ばす。
    actual_steps = len(holdout)
    horizon = max(int(requested_horizon), actual_steps) if 0 < actual_steps <= 5 else int(requested_horizon)
    cal = mcal.get_calendar(prepared["setting"]["calendar"])
    future = cal.schedule(start_date=str((anchor + pd.Timedelta(days=1)).date()),
                          end_date=str((anchor + pd.Timedelta(days=120)).date()))
    if len(future) < horizon:
        raise ValueError("予測先の取引日を十分取得できません。")
    dates = []
    for value in future.index[:horizon]:
        item = pd.Timestamp(value)
        dates.append((item.tz_localize(None) if item.tzinfo is not None else item).normalize())
    hist = input_raw.tail(LOOKBACK)
    updated = dict(prepared)
    updated.update({"raw": input_raw, "hist": hist, "last": anchor, "future_dates": dates,
                    "latest_available": latest_available, "holdout": holdout,
                    "origin_offset": actual_steps, "requested_horizon": int(requested_horizon),
                    "horizon": horizon, "mode": mode, "specified_date": specified_date,
                    "zero_volume": int((hist["volume"] == 0).sum()),
                    "flat_ohlc": int((hist[["open", "high", "low", "close"]].nunique(axis=1) == 1).sum())})
    return updated


def sample_paths(predictor, hist: pd.DataFrame, dates: list[pd.Timestamp]) -> tuple[np.ndarray, int]:
    frame = hist.reset_index(drop=True)
    x_dates = pd.Series(hist.index)
    y_dates = pd.Series(dates)
    with inference_lock():
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        output = predictor.predict_batch(
            df_list=[frame.copy() for _ in range(PATH_COUNT)],
            x_timestamp_list=[x_dates.copy() for _ in range(PATH_COUNT)],
            y_timestamp_list=[y_dates.copy() for _ in range(PATH_COUNT)],
            pred_len=len(dates), T=1.0, top_k=0, top_p=0.9, sample_count=1, verbose=False,
        )
    fields = ["open", "high", "low", "close", "volume"]
    paths = np.stack([result[fields].to_numpy(dtype=float) for result in output])
    if paths.shape != (PATH_COUNT, len(dates), len(fields)) or not np.isfinite(paths).all():
        raise ValueError("予測形式または予測値が不正です。")
    if (paths[:, :, 3] <= 0).any():
        raise ValueError("0以下の予測終値が含まれています。")
    o, h, low, close, vol = (paths[:, :, i] for i in range(5))
    invalid = (h < np.maximum.reduce([o, low, close])) | (low > np.minimum.reduce([o, h, close])) | (vol < 0) | (low <= 0)
    return paths, int(invalid.sum())


def select_origins(raw: pd.DataFrame, horizon: int, requested: int, start: date, end: date) -> list[int]:
    candidates = [i for i in range(LOOKBACK - 1, len(raw) - horizon)
                  if start <= raw.index[i].date() and raw.index[i + horizon].date() <= end]
    # 未来の評価対象が重なりにくいよう、予測ステップ間隔で新しい起点から抽出。
    return sorted(list(reversed(candidates))[::horizon][:requested])


def metrics_for(detail: pd.DataFrame, name: str, filtered: bool = False) -> dict:
    origin = detail["origin_close"].to_numpy(dtype=float)
    actual = detail["actual_close"].to_numpy(dtype=float)
    predicted = detail[f"{name}_mean"].to_numpy(dtype=float)
    actual_direction = np.sign(actual - origin)
    pred_change = 100 * (predicted / origin - 1)
    pred_sign = np.sign(predicted - origin)
    if filtered:
        select = (np.abs(pred_change) >= THRESHOLD_PCT) & (actual_direction != 0)
    else:
        select = actual_direction != 0
    n = int(select.sum())
    up = select & (pred_change >= THRESHOLD_PCT)
    down = select & (pred_change <= -THRESHOLD_PCT)
    ups, downs = int(up.sum()), int(down.sum())
    result = {"model": name, "total_origins": len(detail), "direction_n": n,
              "hits": int(np.sum(select & (pred_sign == actual_direction))),
              "up_n": ups, "up_hits": int(np.sum(up & (actual_direction > 0))),
              "down_n": downs, "down_hits": int(np.sum(down & (actual_direction < 0)))}
    result["accuracy_pct"] = 100 * result["hits"] / n if n else np.nan
    result["up_accuracy_pct"] = 100 * result["up_hits"] / ups if ups else np.nan
    result["down_accuracy_pct"] = 100 * result["down_hits"] / downs if downs else np.nan
    result["always_up_pct"] = 100 * float(np.mean(actual_direction[select] > 0)) if n else np.nan
    result["always_down_pct"] = 100 * float(np.mean(actual_direction[select] < 0)) if n else np.nan
    # 全起点の価格誤差は実績横ばいも含め、閾値付きの誤差は選択された共通対象で評価。
    price_select = select if filtered else np.ones(len(detail), dtype=bool)
    result["mape_pct"] = 100 * float(np.mean(np.abs((predicted[price_select] - actual[price_select]) / actual[price_select]))) if price_select.any() else np.nan
    result["flat_mape_pct"] = 100 * float(np.mean(np.abs((origin[price_select] - actual[price_select]) / actual[price_select]))) if price_select.any() else np.nan
    return result


def fmt_pct(value) -> str:
    return "―" if pd.isna(value) else f"{value:.2f}%"


def daily_mean(paths: np.ndarray, dates: list[pd.Timestamp], anchor: float) -> pd.DataFrame:
    values = paths.mean(axis=0)
    df = pd.DataFrame(values, columns=["始値", "高値", "安値", "終値", "出来高"])
    df.insert(0, "日付目安", [x.strftime("%Y-%m-%d") for x in dates])
    df.insert(0, "先", range(1, len(dates) + 1))
    df["最新終値比(%)"] = (df["終値"] / anchor - 1) * 100
    df["高安幅"] = df["高値"] - df["安値"]
    df["幅率_安値比(%)"] = df["高安幅"] / df["安値"].where(df["安値"] > 0) * 100
    return df[["先", "日付目安", "終値", "最新終値比(%)", "始値", "高値", "安値", "高安幅", "幅率_安値比(%)", "出来高"]]


def forecast_figure(hist: pd.DataFrame, paths_by_model: dict, horizon: int, currency: str, dates: list[pd.Timestamp], actual: pd.DataFrame):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True, sharey=True, constrained_layout=True)
    history = hist["close"].reset_index(drop=True)
    last_close = float(history.iloc[-1])
    view_count = min(16, len(history))
    hx = list(range(-view_count + 1, 1))
    # Align by date, not by row number: calendar estimates and actual data can differ.
    future_axis = sorted(set(pd.Timestamp(x).normalize() for x in dates) | set(pd.Timestamp(x).normalize() for x in actual.index))
    positions = {dt: i + 1 for i, dt in enumerate(future_axis)}
    fx = [0] + [positions[pd.Timestamp(x).normalize()] for x in dates]
    actual_x = [0] + [positions[pd.Timestamp(x).normalize()] for x in actual.index]
    actual_y = np.r_[last_close, actual["close"].to_numpy(dtype=float)]
    for ax, name, color in zip(axes, ("mini", "small"), ("#2563eb", "#bc4a2b")):
        samples = paths_by_model[name][:, :, 3]
        mean = samples.mean(axis=0)
        ax.plot(hx, history.tail(view_count), color="#344054", lw=2, label="History")
        for j in range(PATH_COUNT):
            ax.plot(fx, np.r_[last_close, samples[j]], color=color, alpha=.3, lw=1,
                    label="Sample paths" if j == 0 else "_nolegend_")
        ax.plot(fx, np.r_[last_close, mean], color=color, ls="--", marker="o", lw=2, label="Mean")
        if len(actual):
            ax.plot(actual_x, actual_y, color="#111827", marker="o", lw=2.1, label="Actual (after origin)")
        ma = pd.concat([history, pd.Series(mean)], ignore_index=True).rolling(10, min_periods=10).mean()
        ax.plot(hx, ma.iloc[len(history)-view_count:len(history)], color="#17825d", lw=1.5, label="MA10")
        ax.plot(fx, ma.iloc[len(history)-1:], color="#17825d", ls="--", lw=1.5, label="MA10 with forecast")
        ax.axvline(0, color="#777", ls=":")
        ax.axvspan(0, fx[-1], color="#fff4df", alpha=.55)
        ax.set_title(f"Kronos {name}")
        ax.set_ylabel(f"Adjusted price ({currency})")
        ax.grid(True, linestyle=":", alpha=.5)
        ax.legend(ncol=2, fontsize=8)
    ticks = sorted(set([-15, -10, -5, 0] + fx[1:] + ([actual_x[-1]] if len(actual) else [])))
    axes[-1].set_xticks(ticks)
    axes[-1].set_xlabel("Date-aligned steps (0 = forecast origin)")
    return fig


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")


st.title("Kronos｜mini・small 比較")
st.caption("研究・観察用。投資助言・注文機能ではありません。予測、日付目安、確定済みデータを必ず確認してください。")
st.warning("最新データの遅延、調整済み価格、サンプル数、取引コストと学習期間の重複に注意。小標本の一致率は将来の的中確率ではありません。")

with st.form("kronos_compare_settings"):
    ticker_input = st.text_input("Yahoo Financeティッカー（1銘柄）", value="ES=F", help="先物 ES=F・NQ=F・GC=F・NIY=F、ドル円 JPY=X、株式 SPY・1320.T・0050.TW。N225はCME円建て先物、USDJPYは現物ドル円として扱います。")
    c1, c2 = st.columns(2)
    horizon = c1.selectbox("予測する営業日数", options=[1, 2, 3, 4, 5], index=2)
    backtest_count = c2.selectbox("過去検証の起点数", options=[0, 10, 20, 40, 60], index=0, help="初期値0は予測のみ。40〜60起点はCPUで時間がかかります。")
    today = datetime.now(timezone.utc).date()
    c5, c6 = st.columns(2)
    origin_mode = c5.selectbox("予測の起点", options=[0, 1, 2, 3, 4, 5, -1],
        format_func=lambda x: "最新データ" if x == 0 else ("日付を指定" if x == -1 else f"最新から{x}本前"),
        help="『5本前』なら最新の確定日足から5行前までを入力し、その後の実績と比較します。")
    origin_date = c6.date_input("起点日（日付を指定した場合のみ）", value=today - timedelta(days=7),
        help="休場日・データ欠損日は指定日以前で最も近い日足を起点にします。")
    c3, c4 = st.columns(2)
    start = c3.date_input("検証開始日", value=today - timedelta(days=365))
    end = c4.date_input("検証終了日", value=today)
    run = st.form_submit_button("予測・比較を実行", type="primary")

if run:
    st.session_state.pop("kronos_compare_result", None)
    entered_ticker = ticker_input.strip().upper()
    ticker = ALIASES.get(entered_ticker, entered_ticker)
    try:
        if start > end:
            raise ValueError("検証開始日は終了日以前にしてください。")
        if not ticker:
            raise ValueError("ティッカーを入力してください。")
        with st.spinner("確定済み日足を取得し、モデルを準備しています。初回は重みのダウンロードがあります。"):
            prepared = prepare_data(ticker, int(horizon))
            prepared = set_prediction_origin(prepared, int(origin_mode), origin_date, int(horizon))
            horizon = prepared["horizon"]
            hist = prepared["hist"]
            models = {name: model_for(name) for name in ("mini", "small")}
            latest, invalid = {}, {}
            for name in ("mini", "small"):
                latest[name], invalid[name] = sample_paths(models[name], hist, prepared["future_dates"])
        records, skipped, bt_error = [], [], None
        if backtest_count:
            raw = prepared["raw"]
            origins = select_origins(raw, int(horizon), int(backtest_count), start, end)
            if not origins:
                bt_error = "指定期間に検証可能な起点がありません。期間・データの範囲を確認してください。"
            else:
                progress = st.progress(0, text=f"過去検証を開始：{len(origins)}起点 × 2モデル。CPUでは時間がかかります。")
                for number, i in enumerate(origins, 1):
                    progress.progress((number - 1) / len(origins), text=f"過去検証 {number}/{len(origins)}：{raw.index[i].date()}")
                    history = raw.iloc[i - LOOKBACK + 1:i + 1]
                    future_dates = list(raw.index[i + 1:i + 1 + int(horizon)])
                    try:
                        mini_paths, _ = sample_paths(models["mini"], history, future_dates)
                        small_paths, _ = sample_paths(models["small"], history, future_dates)
                        records.append({"origin_date": raw.index[i].strftime("%Y-%m-%d"),
                                        "target_date": raw.index[i + int(horizon)].strftime("%Y-%m-%d"),
                                        "origin_close": float(history["close"].iloc[-1]),
                                        "actual_close": float(raw["close"].iloc[i + int(horizon)]),
                                        "mini_mean": float(mini_paths[:, -1, 3].mean()),
                                        "small_mean": float(small_paths[:, -1, 3].mean())})
                    except Exception as exc:
                        # 片方だけを集計しない。除外日と理由をユーザーに表示する。
                        skipped.append({"date": str(raw.index[i].date()), "error": str(exc)})
                progress.progress(1.0, text=f"過去検証完了：{len(records)}起点、両モデル共通")
        st.session_state["kronos_compare_result"] = {
            "ticker": ticker, "entered_ticker": entered_ticker, "horizon": int(horizon), "requested": int(backtest_count),
            "start": start, "end": end, "prepared": prepared, "latest": latest,
            "invalid": invalid, "records": records, "skipped": skipped, "bt_error": bt_error,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        st.error(f"処理できませんでした：{exc}")
        st.info("データ遅延・ティッカー・依存ライブラリ・取引日カレンダー・メモリ制限を確認してください。")

result = st.session_state.get("kronos_compare_result")
if result:
    prepared = result["prepared"]
    hist = prepared["hist"]
    setting = prepared["setting"]
    anchor = float(hist["close"].iloc[-1])
    last = prepared["last"]
    dates = prepared["future_dates"]
    ticker = result["ticker"]
    horizon = result["horizon"]
    st.subheader(f"{ticker}｜予測結果")
    st.caption(f"対象: {setting['label']} / 入力最終日: {last.date()} / 入力終値: {anchor:,.2f} {setting['currency']} / 予測先目安: {dates[-1].date()} / 実行時刻UTC: {result['generated_at_utc']}")
    if result["entered_ticker"] != ticker:
        st.info(f"入力 {result['entered_ticker']} は {ticker} として取得しました。")
    if setting.get("kind") == "futures":
        st.warning("先物：Yahoo Financeの日足・取引日ラベルと実際の取引所の取引日・時間は一致しない場合があります。限月切替や連続足の調整方法は未検証です。日付・価格は注文に使わず公式取引所・取引業者の情報を確認してください。")
    elif setting.get("kind") == "forex":
        st.warning("JPY=Xはドル円の現物為替レートで、為替先物ではありません。為替に統一された取引所終値はなく、UTC平日を日付の目安に使用します。")
    elif setting.get("kind") == "index":
        st.info("これは株価指数であり、先物ではありません。")
    if prepared["zero_volume"]:
        st.warning(f"直近{len(hist)}本のうち出来高が0の日が{prepared['zero_volume']}本あります。特に為替は取引所全体の出来高を取得できず、モデルの予測品質に影響する可能性があります。")
    if prepared["flat_ohlc"]:
        message = f"直近{len(hist)}本のうち始値・高値・安値・終値がすべて同じ日が{prepared['flat_ohlc']}本あります。データが十分でない可能性があります。"
        if prepared["flat_ohlc"] / len(hist) >= 0.2:
            st.error(message + " この予測と検証結果の利用は保留し、元データを確認してください。")
        else:
            st.warning(message)
    if prepared["latest_available"] < prepared["last_closed"]:
        st.error(f"データ遅延または売買停止の可能性：最新足 {prepared['latest_available'].date()} / カレンダー上の直近終了日 {prepared['last_closed'].date()}。予測の利用を保留してください。")
    if prepared["mode"] == -1:
        st.info(f"指定日 {prepared['specified_date']} に対し、利用できる直近の日足 {last.date()} を起点にしました。")
    if prepared["origin_offset"]:
        st.info(f"過去起点モード：{last.date()} までのデータで予測。最新の取得済み確定足 {prepared['latest_available'].date()} まで、その後の実績が {prepared['origin_offset']} 本あります。")
        st.caption("起点より後の価格は予測入力と過去検証の入力には使っていません。ただし現在取得した調整済みデータやモデルを使用するため、当時の完全な再現や学習期間から独立した検証ではありません。")
        if prepared["horizon"] != prepared["requested_horizon"]:
            st.info(f"起点からの実績を比較するため、予測日数を {prepared['requested_horizon']} から {prepared['horizon']} に自動延長しました。")
        if prepared["origin_offset"] > prepared["horizon"]:
            st.warning("起点から最新までの実績は5本を超えています。予測は選択した日数（最大5本）までで、それ以降は実績のみ表示します。")
    if ticker.endswith(".TWO"):
        st.info("台湾店頭市場はTWSEカレンダーを近似使用しています。")
    if setting["label"] == "米国":
        st.info("米国市場の日付はNYSEカレンダーを使用しています。NASDAQ等は近似です。")
    overview = []
    for name in ("mini", "small"):
        paths = result["latest"][name]
        final = paths[:, -1, 3]
        changes = 100 * (final / anchor - 1)
        overview.append({"モデル": name, "平均終値": float(final.mean()), "平均変化率(%)": 100 * (float(final.mean()) / anchor - 1),
                         "5本の最小": float(final.min()), "5本の最大": float(final.max()),
                         "上向き": int((changes >= THRESHOLD_PCT).sum()),
                         "下向き": int((changes <= -THRESHOLD_PCT).sum()),
                         "小幅": int(((changes > -THRESHOLD_PCT) & (changes < THRESHOLD_PCT)).sum())})
        if result["invalid"][name]:
            st.warning(f"{name}: 高値・安値・始終値・出来高の不整合が {result['invalid'][name]} 箇所あります。価格を注文条件として利用しないでください。")
    st.dataframe(pd.DataFrame(overview).round(2), hide_index=True, use_container_width=True)
    st.caption("5系列の上向き・下向きはモデル内サンプルの本数。上昇確率ではなく、最小・最大も予測区間ではありません。")
    actual = prepared["holdout"]
    fig = forecast_figure(hist, result["latest"], horizon, setting["currency"], dates, actual)
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
    st.caption("細線：5系列、太い点線：平均、黒線：起点後の実績。未来のMA10は平均予測終値を含む計算値です。0は予測起点。日付が一致しない場合は、欠損を埋めず別の日として扱います。")
    if len(actual):
        st.subheader("起点から最新までの実績")
        observed = pd.concat([hist.tail(1), actual]).copy()
        actual_table = observed.reset_index(drop=True).rename(columns={"open": "実績始値", "high": "実績高値", "low": "実績安値", "close": "実績終値", "volume": "出来高"})
        actual_table.insert(0, "日付", [x.strftime("%Y-%m-%d") for x in observed.index])
        actual_table.insert(0, "起点後の実績本数", range(len(observed)))
        actual_table["起点終値比(%)"] = (actual_table["実績終値"] / anchor - 1) * 100
        st.dataframe(actual_table.round(4), hide_index=True, use_container_width=True)
        st.download_button("起点からの実績CSV", csv_bytes(actual_table), file_name=f"kronos_{ticker.replace('.', '_')}_actual.csv", mime="text/csv")
        st.subheader("日別比較｜予測平均と実績")
        forecast_values = {name: result["latest"][name][:, :, 3].mean(axis=0) for name in ("mini", "small")}
        forecast_by_day = {name: dict(zip(dates, forecast_values[name])) for name in forecast_values}
        compare_rows = []
        for dt in sorted(set(dates) | set(actual.index)):
            realized = float(actual.loc[dt, "close"]) if dt in actual.index else np.nan
            row = {"日付": dt.strftime("%Y-%m-%d"), "実績終値": realized,
                   "実績の起点比(%)": (realized / anchor - 1) * 100 if pd.notna(realized) else np.nan}
            for name in ("mini", "small"):
                prediction = forecast_by_day[name].get(dt, np.nan)
                row[f"{name}予測平均"] = prediction
                row[f"{name}誤差(%)"] = (prediction / realized - 1) * 100 if pd.notna(realized) and pd.notna(prediction) else np.nan
            compare_rows.append(row)
        comparison = pd.DataFrame(compare_rows)
        st.dataframe(comparison.round(4), hide_index=True, use_container_width=True)
        st.caption("同じ日付同士で比較。空欄は実績または予測がない日、誤差は（予測÷実績−1）です。")
        st.download_button("予測と実績の日別比較CSV", csv_bytes(comparison), file_name=f"kronos_{ticker.replace('.', '_')}_comparison.csv", mime="text/csv")
        missing_dates = [dt.strftime("%Y-%m-%d") for dt in dates if dt <= prepared["latest_available"] and dt not in actual.index]
        if missing_dates:
            st.warning("カレンダー上の予測日と実績が一致しない日があります：" + "、".join(missing_dates) + "。取引日・欠損を確認してください。")
    for name in ("mini", "small"):
        st.subheader(f"{name}｜日別の5系列平均")
        daily = daily_mean(result["latest"][name], dates, anchor)
        if len(actual):
            aligned = actual["close"].reindex(pd.DatetimeIndex(dates)).to_numpy(dtype=float)
            daily["実績終値"] = aligned
            daily["予測−実績"] = daily["終値"] - daily["実績終値"]
            daily["予測誤差(%)"] = (daily["終値"] / daily["実績終値"] - 1) * 100
        st.dataframe(daily.round(4), hide_index=True, use_container_width=True)
        st.download_button(f"{name} 日別平均CSV", csv_bytes(daily), file_name=f"kronos_{ticker.replace('.', '_')}_{name}_daily.csv", mime="text/csv", key=f"download_daily_{name}")
    st.caption("高安幅＝予測高値−予測安値、幅率＝高安幅÷予測安値。いずれも実際の取引可能値幅や予測区間ではありません。")
    path_rows = []
    for name in ("mini", "small"):
        for k in range(PATH_COUNT):
            for j, dt in enumerate(dates):
                values = result["latest"][name][k, j]
                path_rows.append({"model": name, "sample": k + 1, "step": j + 1, "date_estimate": dt.strftime("%Y-%m-%d"),
                                  "open": values[0], "high": values[1], "low": values[2], "close": values[3], "volume": values[4]})
    st.download_button("5系列の詳細CSV", csv_bytes(pd.DataFrame(path_rows)), file_name=f"kronos_{ticker.replace('.', '_')}_paths.csv", mime="text/csv")

    st.subheader("過去検証")
    if prepared["origin_offset"]:
        st.caption(f"複数起点の過去検証は、選択した予測起点 {last.date()} までの実績だけを使用します。")
    if not result["requested"]:
        st.info("過去検証は実施していません。モデルの優劣は判断できません。")
    elif result["bt_error"]:
        st.warning(result["bt_error"])
    elif not result["records"]:
        st.error("両モデルで比較できた起点がありません。")
    else:
        detail = pd.DataFrame(result["records"])
        st.caption(f"指定期間: {result['start']}〜{result['end']}。実際の起点: {detail['origin_date'].min()}〜{detail['origin_date'].max()}、最終実績日: {detail['target_date'].max()}。起点と実績日の両方を指定期間内に制限。")
        if len(detail) < result["requested"]:
            st.warning(f"希望 {result['requested']} 起点に対して集計できたのは {len(detail)} 起点です。")
        if result["skipped"]:
            st.warning(f"片方のモデルでエラーが出た {len(result['skipped'])} 起点は、両モデルから除外しました。")
            with st.expander("除外した起点と理由"):
                st.dataframe(pd.DataFrame(result["skipped"]), hide_index=True, use_container_width=True)
        rows_all = []
        rows_filtered = []
        for name in ("mini", "small"):
            all_m = metrics_for(detail, name, filtered=False)
            fil = metrics_for(detail, name, filtered=True)
            rows_all.append({"モデル": name, "起点数": all_m["total_origins"], "方向一致/判定": f"{all_m['hits']}/{all_m['direction_n']}",
                             "方向一致率": fmt_pct(all_m["accuracy_pct"]), "毎回上昇": fmt_pct(all_m["always_up_pct"]),
                             "毎回下落": fmt_pct(all_m["always_down_pct"]), "価格誤差": fmt_pct(all_m["mape_pct"]),
                             "前回終値誤差": fmt_pct(all_m["flat_mape_pct"])})
            rows_filtered.append({"モデル": name, "比較件数": fil["direction_n"], "全体一致": fmt_pct(fil["accuracy_pct"]),
                                  "毎回上昇": fmt_pct(fil["always_up_pct"]), "毎回下落": fmt_pct(fil["always_down_pct"]),
                                  "上向き一致/件数": f"{fil['up_hits']}/{fil['up_n']} ({fmt_pct(fil['up_accuracy_pct'])})",
                                  "下向き一致/件数": f"{fil['down_hits']}/{fil['down_n']} ({fmt_pct(fil['down_accuracy_pct'])})",
                                  "価格誤差": fmt_pct(fil["mape_pct"]), "前回終値誤差": fmt_pct(fil["flat_mape_pct"])})
        st.markdown("**同一起点・同一実績での比較（全起点）**")
        st.dataframe(pd.DataFrame(rows_all), hide_index=True, use_container_width=True)
        st.markdown("**±0.5%以上を予測した起点の方向別比較**")
        st.dataframe(pd.DataFrame(rows_filtered), hide_index=True, use_container_width=True)
        st.caption("方向判定では実績横ばいを除外。閾値付き表の対象起点はモデルごとに異なり得るため、mini・smallの主比較は全起点表を使用。価格誤差は実績終値に対する絶対誤差率の平均（MAPE）。")
        st.warning("検証結果は過去の少数例です。取引コスト・約定時刻を含む売買損益や、独立期間での有効性は確認していません。")
        st.download_button("過去検証の詳細CSV", csv_bytes(detail), file_name=f"kronos_{ticker.replace('.', '_')}_backtest.csv", mime="text/csv")
