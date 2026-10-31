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


def market_for(ticker: str) -> dict:
    if re.fullmatch(r"[A-Z0-9]{4,6}\.T", ticker):
        return {"calendar": "JPX", "tz": "Asia/Tokyo", "currency": "JPY", "label": "日本"}
    if re.fullmatch(r"[A-Z0-9]{4,6}\.(TW|TWO)", ticker):
        return {"calendar": "XTAI", "tz": "Asia/Taipei", "currency": "TWD", "label": "台湾"}
    if re.fullmatch(r"[A-Z][A-Z0-9]{0,9}(?:-[A-Z])?", ticker):
        return {"calendar": "NYSE", "tz": "America/New_York", "currency": "USD", "label": "米国"}
    raise ValueError("対応する形式は SPY / 1320.T / 0050.TW です。日本・台湾は接尾辞まで入力してください。")


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
    return {"raw": raw, "hist": raw.tail(LOOKBACK), "setting": setting, "last": last,
            "last_closed": last_closed, "future_dates": dates, "local_today": today}


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


def forecast_figure(hist: pd.DataFrame, paths_by_model: dict, horizon: int, currency: str):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True, sharey=True, constrained_layout=True)
    history = hist["close"].reset_index(drop=True)
    last_close = float(history.iloc[-1])
    view_count = min(16, len(history))
    hx = list(range(-view_count + 1, 1))
    fx = list(range(horizon + 1))
    for ax, name, color in zip(axes, ("mini", "small"), ("#2563eb", "#bc4a2b")):
        samples = paths_by_model[name][:, :, 3]
        mean = samples.mean(axis=0)
        ax.plot(hx, history.tail(view_count), color="#344054", lw=2, label="History")
        for j in range(PATH_COUNT):
            ax.plot(fx, np.r_[last_close, samples[j]], color=color, alpha=.3, lw=1,
                    label="Sample paths" if j == 0 else "_nolegend_")
        ax.plot(fx, np.r_[last_close, mean], color=color, ls="--", marker="o", lw=2, label="Mean")
        ma = pd.concat([history, pd.Series(mean)], ignore_index=True).rolling(10, min_periods=10).mean()
        ax.plot(hx, ma.iloc[len(history)-view_count:len(history)], color="#17825d", lw=1.5, label="MA10")
        ax.plot(fx, ma.iloc[len(history)-1:], color="#17825d", ls="--", lw=1.5, label="MA10 with forecast")
        ax.axvline(0, color="#777", ls=":")
        ax.axvspan(0, horizon, color="#fff4df", alpha=.55)
        ax.set_title(f"Kronos {name}")
        ax.set_ylabel(f"Adjusted price ({currency})")
        ax.grid(True, linestyle=":", alpha=.5)
        ax.legend(ncol=2, fontsize=8)
    axes[-1].set_xticks([-15, -10, -5, 0] + list(range(1, horizon + 1)))
    axes[-1].set_xlabel("Trading steps (0 = latest input)")
    return fig


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")


st.title("Kronos｜mini・small 比較")
st.caption("研究・観察用。投資助言・注文機能ではありません。予測、日付目安、確定済みデータを必ず確認してください。")
st.warning("最新データの遅延、調整済み価格、サンプル数、取引コストと学習期間の重複に注意。小標本の一致率は将来の的中確率ではありません。")

with st.form("kronos_compare_settings"):
    ticker_input = st.text_input("Yahoo Financeティッカー（1銘柄）", value="SPY", help="米国 SPY、日本 1320.T、台湾 0050.TW。指数・暗号資産などは対象外。")
    c1, c2 = st.columns(2)
    horizon = c1.selectbox("予測する営業日数", options=[1, 2, 3, 4, 5], index=2)
    backtest_count = c2.selectbox("過去検証の起点数", options=[0, 10, 20, 40, 60], index=0, help="初期値0は予測のみ。40〜60起点はCPUで時間がかかります。")
    today = datetime.now(timezone.utc).date()
    c3, c4 = st.columns(2)
    start = c3.date_input("検証開始日", value=today - timedelta(days=365))
    end = c4.date_input("検証終了日", value=today)
    run = st.form_submit_button("予測・比較を実行", type="primary")

if run:
    st.session_state.pop("kronos_compare_result", None)
    ticker = ticker_input.strip().upper()
    try:
        if start > end:
            raise ValueError("検証開始日は終了日以前にしてください。")
        if not ticker:
            raise ValueError("ティッカーを入力してください。")
        with st.spinner("確定済み日足を取得し、モデルを準備しています。初回は重みのダウンロードがあります。"):
            prepared = prepare_data(ticker, int(horizon))
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
            "ticker": ticker, "horizon": int(horizon), "requested": int(backtest_count),
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
    st.caption(f"入力最終日: {last.date()} / 調整済み終値: {anchor:,.2f} {setting['currency']} / 予測先目安: {dates[-1].date()} / 実行時刻UTC: {result['generated_at_utc']}")
    if last < prepared["last_closed"]:
        st.error(f"データ遅延または売買停止の可能性：最新足 {last.date()} / カレンダー上の直近終了日 {prepared['last_closed'].date()}。予測の利用を保留してください。")
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
    fig = forecast_figure(hist, result["latest"], horizon, setting["currency"])
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)
    st.caption("細線：5系列、太い点線：平均。未来のMA10は平均予測終値を含む計算値です。0は各銘柄の最新入力日。")
    for name in ("mini", "small"):
        st.subheader(f"{name}｜日別の5系列平均")
        daily = daily_mean(result["latest"][name], dates, anchor)
        st.dataframe(daily.round(2), hide_index=True, use_container_width=True)
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
