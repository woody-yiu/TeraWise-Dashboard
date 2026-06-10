
import os
import json
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import warnings
from finlab import login, data
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import requests

warnings.filterwarnings('ignore')

# 1. Login
token = os.environ.get("FINLAB_TOKEN", "97Y21Yf07Tokqp6rnUxsQKHbc4j+HosTsqE5DNh2oWLA9n+pxaCibJSKUK190ocZ#vip_m")
login(token)

# 2. Parameters
N_DAYS = 5
MIN_LIQ_PCT = 0.6
TOP_GROUPS = 20           
TOP_STOCKS = 50           
PORTFOLIO_SIZE = 15       

STOP_LOSS_PCT = 0.10
FIXED_HOLDING_DAYS = 40
COOLING_OFF_DAYS = 5

# Weights
weights = {"ret": 1.5, "turnover": 1.0, "inst": 1.0, "conc": 1.0}

print("正在抓取並對齊資料...")
close = data.get("price:收盤價")
open_ = data.get("price:開盤價")
volume = data.get("price:成交股數")
benchmark = data.get('taiex_total_index:收盤指數')
benchmark = benchmark[~benchmark.index.duplicated(keep='first')]
benchmark_ma200 = benchmark.rolling(200).mean()

# --- 1. 處理「多重標籤」 (Multi-Label Mapping) ---
print("處理多重題材標籤 (混合動態版：早期回填 + 近期逐日更新)...")
theme_raw = data.get("security_industry_themes").copy()
cat_raw = data.get("security_categories").copy()

# 轉換日期格式
theme_raw['key_date'] = pd.to_datetime(theme_raw['key_date']).dt.normalize()

# 💎 核心魔法：兩段式時空切換 (歷史凍結 -> 現實校正 -> 未來動態)
# 1. 歷史凍結區：擷取 2026-01-01 快照，時空穿越到 2010 年。
# 作用：讓 2012 ~ 2026-04-26 的回測，永遠鎖死在 1/1 的標籤狀態。
snapshot_1_date = pd.to_datetime('2026-01-01')
snapshot_1 = theme_raw[theme_raw['key_date'] <= snapshot_1_date].sort_values('key_date').drop_duplicates('stock_id', keep='last').copy()
snapshot_1['key_date'] = pd.to_datetime('2010-01-01')

# 2. 現實校正點：擷取今天 (2026-04-26) 的「真實最新狀態」，設定在 2026-04-27 觸發。
# 作用：當迴圈跑到 4/27，矩陣會瞬間覆蓋成 Finlab 真正的最新狀態 (把 1/1~4/26 之間所有 Finlab 的隱藏更新一次補齊)。
snapshot_2_date = pd.to_datetime('2026-04-25')
snapshot_2 = theme_raw[theme_raw['key_date'] <= snapshot_2_date].sort_values('key_date').drop_duplicates('stock_id', keep='last').copy()
snapshot_2['key_date'] = pd.to_datetime('2026-04-26')

# 3. 未來動態區：保留今天之後的新更新。
# 作用：4/27 之後的每一天，只要 Finlab 有發佈新標籤，就逐日動態替換。
future_updates = theme_raw[theme_raw['key_date'] > snapshot_2_date].copy()

# 合併三段時空並排序
theme_raw = pd.concat([snapshot_1, snapshot_2, future_updates]).sort_values('key_date')


# 準備名稱與顯示用對照表 (維持最新標籤僅供最終 Dashboard 顯示用)
name_mapper = theme_raw.drop_duplicates("stock_id", keep="last").set_index("stock_id")["name"]
cat_mapper = cat_raw.drop_duplicates("stock_id", keep="last").set_index("stock_id")["category"]

# 提取所有曾經出現過的股票，確立基礎 Universe
theme_all_stocks = theme_raw['stock_id'].unique()

# 解析字串為 List
def parse_themes(x):
    try:
        if isinstance(x, str):
            clean_str = x.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
            return [t.strip() for t in clean_str.split(",") if t.strip() != ""]
        return []
    except:
        return []

theme_raw['theme_list'] = theme_raw['category'].apply(parse_themes)
all_themes = theme_raw['theme_list'].explode().dropna().unique()
all_themes = [t for t in all_themes if t != ""]

# 3. 對齊所有資料的欄位 (寬鬆模式: 以收盤價為主)
print("資料對齊與處理...")
foreign = data.get('institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)')
trust = data.get('institutional_investors_trading_summary:投信買賣超股數')
dealer = data.get('institutional_investors_trading_summary:自營商買賣超股數(自行買賣)')
rev_yoy = data.get("monthly_revenue:去年同月增減(%)")

# 定義核心 Universe
common_cols = (close.columns
               .intersection(volume.columns)
               .intersection(theme_all_stocks)
               .intersection(cat_mapper.index)) 

# 加入 adj=True 確保股價有還原
close = data.get("price:收盤價")[common_cols]
open_ = data.get("price:開盤價")[common_cols]
volume = volume[common_cols]
cat_mapper = cat_mapper[common_cols]
name_mapper = name_mapper.reindex(common_cols).fillna("") 

# 寬鬆處理其他數據
trust = trust.reindex(columns=common_cols, fill_value=0)
dealer = dealer.reindex(columns=common_cols, fill_value=0)
foreign = foreign.reindex(columns=common_cols, fill_value=0) 
rev_yoy = rev_yoy.reindex(columns=common_cols)
rev_yoy = rev_yoy.reindex(close.index, method='ffill')

inst_total = (trust + dealer).reindex(close.index)
inst_buy_yday = inst_total.shift(2)
inst_concentration = inst_total.shift(2) / volume.replace(0, np.nan)
# 1. 舊版邏輯 (負責維持 2026-05-12 以前的歷史紀錄完全不變)
ma200_old = close.rolling(200).mean()
# 2. 改良版邏輯 (修復國巨等股票減資停牌造成 nan 的問題)
ma200_new = close.ffill().rolling(200, min_periods=150).mean()
# 3. 結合兩者：以舊版為基底，只將 5/13 之後的資料替換為改良版
ma200 = ma200_old.copy()
ma200.loc[ma200.index >= '2026-05-13'] = ma200_new.loc[ma200_new.index >= '2026-05-13']

print("計算「動態多重標籤」產業選股訊號 (逐日推進矩陣)...")
ret = close.pct_change(N_DAYS)
turnover = close * volume

selected_stocks_signal = {}
valid_index = close.index.intersection(inst_buy_yday.index)

# 建立動態更新的 Theme Matrix (起初全為 0)
current_theme_matrix = pd.DataFrame(0, index=common_cols, columns=all_themes)
theme_update_idx = 0
theme_raw_records = theme_raw[['key_date', 'stock_id', 'theme_list']].to_dict('records')

for date in valid_index[valid_index >= '2011-12-01']:
    
    # 1. 推進日期：更新當天的標籤
    while theme_update_idx < len(theme_raw_records) and theme_raw_records[theme_update_idx]['key_date'] <= date:
        rec = theme_raw_records[theme_update_idx]
        sid = rec['stock_id']
        if sid in current_theme_matrix.index:
            current_theme_matrix.loc[sid] = 0 # 清除該股票舊標籤
            for t in rec['theme_list']:       # 貼上新標籤
                if t in current_theme_matrix.columns:
                    current_theme_matrix.at[sid, t] = 1
        theme_update_idx += 1

    # 防呆：如果全市場沒標籤就跳過
    if current_theme_matrix.sum().sum() == 0:
        continue
        
    # 2. 計算「當天」的產業指標 (利用矩陣相乘實現極速運算)
    day_ret = ret.loc[date].fillna(0)
    day_turnover = turnover.loc[date].fillna(0)
    day_inst = inst_buy_yday.loc[date].fillna(0)
    day_conc = inst_concentration.loc[date].fillna(0)
    
    # 💎 防護機制：剔除當下還沒有任何股票的「未來幽靈產業」，確保排名分母與歷史完全一致
    active_themes = current_theme_matrix.columns[current_theme_matrix.sum(axis=0) > 0]
    active_matrix = current_theme_matrix[active_themes]
    
    g_sum_ret = day_ret @ active_matrix
    g_sum_turnover = day_turnover @ active_matrix
    g_sum_inst = day_inst @ active_matrix
    g_sum_conc = day_conc @ active_matrix
    
    # 計算有有效資料的股票數量以求平均
    has_data_ret = (~ret.loc[date].isna()).astype(int)
    has_data_conc = (~inst_concentration.loc[date].isna()).astype(int)
    g_count_ret = has_data_ret @ active_matrix
    g_count_conc = has_data_conc @ active_matrix
    
    g_mean_ret = g_sum_ret / g_count_ret.replace(0, np.nan)
    g_mean_conc = g_sum_conc / g_count_conc.replace(0, np.nan)
    
    # 3. 結算當天產業評分
    g_score = (g_mean_ret.rank(pct=True) * weights["ret"] +
               g_sum_turnover.rank(pct=True) * weights["turnover"] +
               g_sum_inst.rank(pct=True) * weights["inst"] +
               g_mean_conc.rank(pct=True) * weights["conc"])
    
    # 找出前 TOP_GROUPS 個強勢產業
    is_top = g_score.rank(ascending=False) <= TOP_GROUPS
    strong_themes = is_top[is_top].index.tolist()

    
    if not strong_themes: continue
    
    # 4. 選出候選股票
    candidates = current_theme_matrix[strong_themes].sum(axis=1)
    stocks_in_groups = candidates[candidates > 0].index.tolist()
    
    try:
        df = pd.DataFrame({
            "ret": ret.loc[date, stocks_in_groups],
            "turnover": turnover.loc[date, stocks_in_groups],
            "inst": inst_buy_yday.loc[date, stocks_in_groups],
            "conc": inst_concentration.loc[date, stocks_in_groups],
            "yoy": rev_yoy.loc[date, stocks_in_groups]
        }).dropna()
    except: continue
    
    if df.empty: continue
    liq_cut = df["turnover"].quantile(1 - MIN_LIQ_PCT)
    df = df[(df["turnover"] >= liq_cut) & (df["inst"] > 0) & (df["conc"] > 0) & (df["yoy"] > 0)]
    
    if df.empty: continue
    df["score"] = (df["ret"].rank(pct=True) * weights["ret"] +
                  df["turnover"].rank(pct=True) * weights["turnover"] +
                  df["inst"].rank(pct=True) * weights["inst"] +
                  df["conc"].rank(pct=True) * weights["conc"])
    
    # MA200 濾網
    current_ma200 = ma200.loc[date]
    current_close = close.loc[date]
    valid_stocks = df.index.intersection(current_ma200.index).intersection(current_close.index)
    df = df.loc[valid_stocks]
    df = df[current_close[df.index] > current_ma200[df.index]]
    
    selected_stocks_signal[date] = df.sort_values("score", ascending=False).head(TOP_STOCKS).index.tolist()

print("訊號計算完成。")

# --- 終極實戰時序版：收盤判定，次日開盤賣出 ---
print(f"執行回測 (目標持股 {PORTFOLIO_SIZE} 檔，僅限每月 10-15 號進場)...【修正版：次日開盤賣出】")

INITIAL_CAPITAL = 10_000_000
CASH = INITIAL_CAPITAL
PORTFOLIO = []         # 當前持倉
PENDING_EXITS = []     # 標記為「待賣出」的股票列表
TRADE_LOG = []
NAV_HISTORY = []
last_buy_date = {}

backtest_dates = close.index.intersection(valid_index)
backtest_dates = backtest_dates[backtest_dates >= '2012-01-01']

# Fix: Fill NaNs for valuation purposes
close_ffill = close.ffill()

for i, today in enumerate(backtest_dates):
    yesterday = backtest_dates[i-1] if i > 0 else None
    
    # --- [1] 上午開盤：處理昨晚被標記的「待賣出」股票 ---
    failed_exits = []
    for p in PENDING_EXITS:
        sell_price = open_.at[today, p['stock_id']]
        if pd.isna(sell_price): 
            sell_price = close.at[today, p['stock_id']] # 若無開盤價則以今日收盤代替
            
        # 防呆：若價格仍為 NaN 則跳過
        if pd.isna(sell_price):
            failed_exits.append(p)
            continue

        revenue = sell_price * p['shares']
        fee = revenue * (0.001425 * 0.1 + 0.003) 
        CASH += (revenue - fee)
        
        TRADE_LOG.append({
            'stock_id': p['stock_id'], 'entry_date': p['entry_date'], 'exit_date': today,
            'entry_price': round(p['entry_price'], 2), 'exit_price': round(sell_price, 2),
            'ret': (revenue - fee - p['cost']) / p['cost'], 'exit_reason': p['reason']
        })
    
    PENDING_EXITS = [] # 執行完畢，清空待賣清單，此時空位才真正釋放
    if failed_exits:
        PORTFOLIO.extend(failed_exits)
    
    # --- [2] 上午開盤：進場邏輯 (檢查空位併補貨) ---
    if yesterday is not None:
        market_pass = True
        bm_yesterday = benchmark.at[yesterday, benchmark.columns[0]] if yesterday in benchmark.index else np.nan
        bm_ma_yesterday = benchmark_ma200.at[yesterday, benchmark_ma200.columns[0]] if yesterday in benchmark_ma200.index else np.nan
        if pd.notna(bm_yesterday) and pd.notna(bm_ma_yesterday) and bm_yesterday < bm_ma_yesterday: 
            market_pass = False
        
        is_entry_window = 10 <= today.day <= 15
                
        if market_pass and is_entry_window:
            slots_to_fill = PORTFOLIO_SIZE - len(PORTFOLIO)
            if slots_to_fill > 0:
                signals = selected_stocks_signal.get(yesterday, [])
                for sid in signals:
                    if slots_to_fill <= 0: break
                    if any(p['stock_id'] == sid for p in PORTFOLIO): continue
                    if pd.isna(open_.at[today, sid]): continue
                    
                    curr_ma = ma200.at[yesterday, sid]
                    curr_close = close.at[yesterday, sid]
                    if pd.isna(curr_ma) or curr_close <= curr_ma: 
                        continue
                    
                    lbd = last_buy_date.get(sid)
                    if lbd and (close.index.get_loc(yesterday) - close.index.get_loc(lbd)) <= COOLING_OFF_DAYS: 
                        continue
                    
                    holdings_val = sum(close.at[today, pp['stock_id']] * pp['shares'] if pd.notna(close.at[today, pp['stock_id']]) else pp['entry_price'] * pp['shares'] for pp in PORTFOLIO)
                    
                    if pd.isna(CASH): CASH = 0
                    target_value = min((CASH + holdings_val) / PORTFOLIO_SIZE, CASH * 0.98)
                    entry_price = open_.at[today, sid]
                    
                    # 買入計算防呆 (開放零股)
                    cost_per_share = entry_price * (1 + 0.001425 * 0.1)
                    if pd.isna(target_value) or pd.isna(cost_per_share) or cost_per_share == 0:
                        continue

                    # 計算能買的「股數 (Shares)」
                    shares = int(target_value / cost_per_share)
                    
                    if shares > 0:
                        cost = shares * cost_per_share
                        CASH -= cost
                        PORTFOLIO.append({
                            'stock_id': sid, 'entry_date': today, 'entry_price': entry_price, 
                            'shares': shares, 'cost': cost, 'entry_idx': i
                        })
                        last_buy_date[sid] = today
                        slots_to_fill -= 1

    # --- [3] 下午收盤：結算淨值與判定明日賣出清單 ---
    # Use close_ffill to handle suspended stocks
    current_holdings_value = sum(close_ffill.at[today, p['stock_id']] * p['shares'] for p in PORTFOLIO)
    current_nav = CASH + current_holdings_value
    
    new_active_portfolio = []
    for p in PORTFOLIO:
        curr_price = close.at[today, p['stock_id']]
        exit_reason = None
        if pd.notna(curr_price):
            if curr_price < p['entry_price'] * (1 - STOP_LOSS_PCT): 
                exit_reason = "Stop Loss"
            elif (i - p['entry_idx']) >= FIXED_HOLDING_DAYS: 
                exit_reason = "Time Exit"
        if exit_reason:
            p['reason'] = exit_reason
            PENDING_EXITS.append(p)
        else:
            new_active_portfolio.append(p)
    PORTFOLIO = new_active_portfolio

    NAV_HISTORY.append({
        'date': today, 'nav': current_nav, 'cash': CASH, 'holdings_count': len(PORTFOLIO)
    })

df_nav = pd.DataFrame(NAV_HISTORY).set_index('date')
print("✅ 實戰模型修正完成：今日收盤觸發，次日開盤賣出。")

df_nav['return'] = df_nav['nav'].pct_change()
df_nav['peak'] = df_nav['nav'].cummax()
df_nav['drawdown'] = (df_nav['nav'] - df_nav['peak']) / df_nav['peak']

# --- 終極自動化：數據分析 + 訊號補回 + 雲端發布 ---
print(f"🚀 啟動 TeraWise 雲端發布流程 [{datetime.now().strftime('%Y-%m-%d %H:%M')}]...")

# 1. 基礎指標計算
last_data_date = df_nav.index[-1].strftime('%Y-%m-%d')
last_prices = close.iloc[-1]

daily_ret = df_nav['nav'].pct_change().dropna()
total_days = (df_nav.index[-1] - df_nav.index[0]).days
if total_days < 1: total_days = 1

# 2. 年化報酬 (CAGR)
ann_ret = (df_nav['nav'].iloc[-1] / df_nav['nav'].iloc[0]) ** (365 / total_days) - 1

# 3. 年化波動率
ann_vol = daily_ret.std() * np.sqrt(252)

# 4. 夏普比率 (Sharpe Ratio)
risk_free_rate = 0.02 
ann_arithmetic_mean = daily_ret.mean() * 252
sharpe = (ann_arithmetic_mean - risk_free_rate) / ann_vol if ann_vol != 0 else 0

# 5. 其他指標
ann_downside_deviation = np.sqrt((daily_ret.clip(upper=0)**2).mean()) * np.sqrt(252)
mdd_val = abs(df_nav['drawdown'].min())
calmar = ann_ret / mdd_val if mdd_val != 0 else 0
downside_ret = daily_ret[daily_ret < 0]
ann_downside_vol = downside_ret.std() * np.sqrt(252)
sortino = (ann_ret - risk_free_rate) / ann_downside_vol if ann_downside_vol != 0 else 0

# 2. 交易勝報比統計
trade_stats = {"win_rate": 0, "avg_win": 0, "avg_loss": 0, "profit_factor": 0, "total_trades": 0}
if TRADE_LOG:
    rets = [t['ret'] for t in TRADE_LOG]
    wins = [r for r in rets if r > 0]; losses = [r for r in rets if r <= 0]
    trade_stats = {
        "win_rate": round(len(wins)/len(rets)*100, 2), 
        "avg_win": round(np.mean(wins)*100, 2) if wins else 0,
        "avg_loss": round(np.mean(losses)*100, 2) if losses else 0, 
        "profit_factor": round(sum(wins)/abs(sum(losses)), 2) if losses and sum(losses)!=0 else 0,
        "total_trades": len(rets)
    }

# 3. 處理持倉細節
curr_holdings_data = []
sector_counter = {}
all_holdings = PORTFOLIO + PENDING_EXITS
if all_holdings:
    for p in all_holdings:
        sid = p['stock_id']
        name = name_mapper.get(sid, sid)
        cat = cat_mapper.get(sid, "其他")
        sector_counter[cat] = sector_counter.get(cat, 0) + 1
        pnl = (last_prices[sid] / p['entry_price'] - 1)
        
        days_held = len(backtest_dates) - 1 - p['entry_idx']
        remaining = FIXED_HOLDING_DAYS - days_held
        expected_exit = backtest_dates[-1] + pd.tseries.offsets.BusinessDay(remaining)
        
        curr_holdings_data.append({
            "stock_id": sid, "name": f"{name}", "category": cat,
            "entry_date": p['entry_date'].strftime('%Y-%m-%d'),
            "expect_exit": expected_exit.strftime('%Y-%m-%d'), 
            "entry_price": round(p['entry_price'], 2), 
            "current_price": round(last_prices[sid], 2),
            "current_date": last_data_date,
            "pnl": round(pnl * 100, 2)
        })
sector_pie = [{"name": k, "value": v} for k, v in sector_counter.items()]

# 4. 補回最近 5 日選股訊號
recent_signals_data = []
if 'selected_stocks_signal' in locals() and selected_stocks_signal:
    signal_dates = sorted(selected_stocks_signal.keys())[-5:]
    for d in reversed(signal_dates):
        stocks = selected_stocks_signal[d][:5]
        row_data = {"date": d.strftime("%Y-%m-%d"), "stocks": []}
        for i, sid in enumerate(stocks):
            name = name_mapper.get(sid, "")
            cat = cat_mapper.get(sid, "Unknown")
            row_data["stocks"].append(f"{i+1}. {sid} {name} ({cat})")
        recent_signals_data.append(row_data)

# 5. 產生「近期基金操作日誌」
recent_ops = []
seen_buys = set()
if TRADE_LOG:
    for t in TRADE_LOG:
        name = name_mapper.get(t['stock_id'], str(t['stock_id']))
        recent_ops.append({
            "date": t['exit_date'].strftime('%Y-%m-%d'), "action": "賣出",
            "stock_id": t['stock_id'], "name": f"{name}",
            "price": round(t['exit_price'], 2), "reason": t['exit_reason'],
            "pnl": round(t['ret'] * 100, 2),
            "entry_info": f"({t['entry_date'].strftime('%m/%d')} 以 {round(t['entry_price'], 2)} 買入)"
        })
        key = (t['stock_id'], t['entry_date'])
        if key not in seen_buys:
            recent_ops.append({
                "date": t['entry_date'].strftime('%Y-%m-%d'), "action": "買入",
                "stock_id": t['stock_id'], "name": f"{name}",
                "price": round(t['entry_price'], 2), "reason": "訊號進場",
                "pnl": "-", "entry_info": ""
            })
            seen_buys.add(key)
if PORTFOLIO:
    for p in PORTFOLIO:
        key = (p['stock_id'], p['entry_date'])
        if key not in seen_buys:
            name = name_mapper.get(p['stock_id'], str(p['stock_id']))
            recent_ops.append({
                "date": p['entry_date'].strftime('%Y-%m-%d'), "action": "買入",
                "stock_id": p['stock_id'], "name": f"{name}",
                "price": round(p['entry_price'], 2), "reason": "訊號進場",
                "pnl": "-", "entry_info": ""
            })
            seen_buys.add(key)
recent_ops.sort(key=lambda x: x['date'], reverse=True)

# 6. 熱圖分析
temp_trades = []
if TRADE_LOG:
    for t in TRADE_LOG:
        temp_trades.append({'sid': t['stock_id'], 'in': pd.to_datetime(t['entry_date']), 'out': pd.to_datetime(t['exit_date'])})
if PORTFOLIO:
    for p in PORTFOLIO:
        temp_trades.append({'sid': p['stock_id'], 'in': pd.to_datetime(p['entry_date']), 'out': pd.to_datetime('2099-12-31')})
df_full_t = pd.DataFrame(temp_trades)
sector_series = pd.Series(cat_mapper)
major_sectors = sector_series.value_counts()[sector_series.value_counts() >= 10].index
monthly_groups = df_nav.resample('M') 
heatmap_data = {}
prev_nav = None
for date, group in monthly_groups:
    if len(group) < 1: continue
    yr, mo = str(date.year), date.month
    start_nav = prev_nav if prev_nav is not None else group['nav'].iloc[0]
    m_ret = (group['nav'].iloc[-1] / start_nav - 1) * 100
    prev_nav = group['nav'].iloc[-1]
    test_day = group.index[-1]
    m_stock_rets = (close.loc[test_day] / close.loc[group.index[0]] - 1)
    val_s = m_stock_rets.index.intersection(cat_mapper.keys())
    m_perf = m_stock_rets[val_s].groupby(cat_mapper).median()
    m_major = m_perf[m_perf.index.isin(major_sectors)]
    market_top = m_major.idxmax() if not m_major.empty else "N/A"
    port_top = "現金"
    if not df_full_t.empty:
        active = df_full_t[(df_full_t['in'] <= test_day) & (df_full_t['out'] >= test_day)]
        if not active.empty:
            possible = active['sid'].map(cat_mapper).value_counts()
            if not possible.empty:
                port_top = possible.idxmax()
    if yr not in heatmap_data: heatmap_data[yr] = {}
    heatmap_data[yr][mo] = {"ret": round(m_ret, 2), "market_top": market_top, "port_top": port_top}

# 7. 歷史平倉明細
historical_trades = []
if TRADE_LOG:
    for t in sorted(TRADE_LOG, key=lambda x: x['exit_date'], reverse=True)[:50]:
        name = name_mapper.get(t['stock_id'], str(t['stock_id']))
        historical_trades.append({
            "stock_id": t['stock_id'], "name": f"{name}",
            "category": cat_mapper.get(t['stock_id'], "其他"), "entry_date": t['entry_date'].strftime('%Y-%m-%d'),
            "exit_date": t['exit_date'].strftime('%Y-%m-%d'), "entry_price": round(t['entry_price'], 2),
            "exit_price": round(t['exit_price'], 2), "ret": round(t['ret'], 4), "exit_reason": t['exit_reason']
        })

# 8. 建構封包並發布
bm_aligned = benchmark.reindex(df_nav.index).ffill()
bm_col = bm_aligned.columns[0]

# Time adjustment for display
tz = timezone(timedelta(hours=8))
last_update_str = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

dashboard_data = {
    "summary": { 
        "last_update": last_update_str, 
        "sharpe": round(sharpe, 2), 
        "sortino": round(sortino, 2),
        "downside_risk": round(ann_downside_deviation * 100, 2), 
        "calmar": round(calmar, 2), 
        "ann_ret": round(ann_ret * 100, 2) 
    },
    "trade_stats": trade_stats,
    "current_holdings": curr_holdings_data,
    "recent_signals": recent_signals_data,
    "trades": historical_trades,
    "operations": recent_ops[:50],
    "sectors": sector_pie,
    "heatmap": heatmap_data,
    "history": [{ "date": d.strftime("%Y-%m-%d"), "nav": round(v, 2), "benchmark": round(bm_aligned.at[d, bm_col], 2), "mdd": round(m * 100, 2) } for d, v, m in zip(df_nav.index, df_nav['nav'], df_nav['drawdown']) ]
}

# --- Telegram 異動通知 ---
print("檢查是否需要發送 Telegram 通知...")
try:
    tg_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    tg_chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if tg_token and tg_chat_id:
        # HTML 安全字元轉義函數，避免特殊字元 (&, <, >) 導致 Telegram API 報錯
        def escape_html(text):
            return str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        # 1. 整理明天要賣出的名單
        sell_msgs = []
        if PENDING_EXITS:
            for p in PENDING_EXITS:
                sid = p['stock_id']
                name = escape_html(name_mapper.get(sid, str(sid)))
                reason = "停損出場" if p.get('reason') == "Stop Loss" else "時間出場"
                sell_msgs.append(f"• {sid} {name} ({reason})")
                
        # 2. 整理明天準備買入的名單
        buy_msgs = []
        last_date = df_nav.index[-1]
        # 推算下一個營業日 (跳過週末，確保不會把週六/日當成下一交易日)
        tomorrow = last_date + timedelta(days=1)
        while tomorrow.weekday() >= 5:  # 5=Saturday, 6=Sunday
            tomorrow = tomorrow + timedelta(days=1)
        
        # 取出最新的大盤日期以供判斷
        latest_bm_date = last_date
        if last_date not in benchmark.index:
            latest_bm_date = benchmark.index[-1] if not benchmark.empty else last_date

        # 模擬大盤濾網檢查 (最新大盤收盤是否站上 200MA)
        market_pass = True
        if latest_bm_date in benchmark.index and latest_bm_date in benchmark_ma200.index:
            bm_today = benchmark.at[latest_bm_date, benchmark.columns[0]]
            bm_ma_today = benchmark_ma200.at[latest_bm_date, benchmark_ma200.columns[0]]
            if pd.notna(bm_today) and pd.notna(bm_ma_today) and bm_today < bm_ma_today: 
                market_pass = False
        
        # 檢查明天是否為建倉日 10-15 號，且大盤在 200MA 之上
        tomorrow_day = tomorrow.day
        if 10 <= tomorrow_day <= 15 and market_pass:
            slots_to_fill = PORTFOLIO_SIZE - len(PORTFOLIO)
            if slots_to_fill > 0:
                signals = selected_stocks_signal.get(last_date, [])
                for sid in signals:
                    if slots_to_fill <= 0: break
                    if any(p['stock_id'] == sid for p in PORTFOLIO): continue
                    
                    # 模擬次日買入的過濾條件
                    curr_ma = ma200.at[last_date, sid]
                    curr_close = close.at[last_date, sid]
                    if pd.isna(curr_ma) or curr_close <= curr_ma: 
                        continue
                        
                    lbd = last_buy_date.get(sid)
                    if lbd and (close.index.get_loc(last_date) - close.index.get_loc(lbd)) <= COOLING_OFF_DAYS: 
                        continue
                        
                    name = escape_html(name_mapper.get(sid, str(sid)))
                    buy_msgs.append(f"• {sid} {name}")
                    slots_to_fill -= 1

        # 3. 整理今日前五名選股訊號
        top5_msgs = []
        signals_today = selected_stocks_signal.get(last_date, [])[:5]
        for i, sid in enumerate(signals_today):
            name = escape_html(name_mapper.get(sid, str(sid)))
            cat = escape_html(cat_mapper.get(sid, "其他"))
            top5_msgs.append(f"{i+1}. {sid} {name} ({cat})")

        # 防呆機制：使用 Firebase 檢查今天是否已發過通知 (本機檔案無法跨 GitHub Actions 保留)
        notify_db = None
        today_str = last_date.strftime('%Y-%m-%d')
        already_notified = False
        try:
            firebase_cert_tg = os.environ.get("FIREBASE_SERVICE_ACCOUNT")
            if firebase_cert_tg:
                cert_dict_tg = json.loads(firebase_cert_tg)
                if not firebase_admin._apps:
                    cred_tg = credentials.Certificate(cert_dict_tg)
                    firebase_admin.initialize_app(cred_tg)
                notify_db = firestore.client()
                notify_doc = notify_db.collection("quant_fund").document("notify_metadata").get()
                if notify_doc.exists and notify_doc.to_dict().get("last_notify_date") == today_str:
                    already_notified = True
        except Exception as e_notify_read:
            print(f"⚠️ 無法讀取 Firebase 通知記錄: {e_notify_read}")

        # 4. 放寬資料同步限制：如果大盤指數尚未更新至今日，則以「最近一個交易日」的大盤狀態為準
        # 避免因為 Finlab 大盤資料延遲而導致完全發不出通知 (包含停損)
        is_data_fully_synced = True
        latest_bm_date = last_date
        if last_date not in benchmark.index:
            is_data_fully_synced = False
            latest_bm_date = benchmark.index[-1] if not benchmark.empty else last_date
            print(f"🕒 個股已更新至 {last_date.strftime('%Y-%m-%d')}，但大盤指數僅至 {latest_bm_date.strftime('%Y-%m-%d')}。將使用最近一天大盤資料作為參考，並照常發送通知。")

        # 5. 發送通知 (放寬限制：只要有異動就發送，不再被 is_data_fully_synced 卡死)
        if (sell_msgs or buy_msgs or top5_msgs) and not already_notified:
            msg = f"📊 <b>【量化策略每日報告】</b>\n📅 資料日期: {last_date.strftime('%Y-%m-%d')}\n"
            
            if top5_msgs:
                msg += "\n🏆 <b>今日策略前五強:</b>\n" + "\n".join(top5_msgs) + "\n"
                
            msg += "\n⚡ <b>明日預計交易異動:</b>\n"
            if sell_msgs or buy_msgs:
                if sell_msgs:
                    msg += "🔴 <b>準備賣出:</b>\n" + "\n".join(sell_msgs) + "\n"
                if buy_msgs:
                    msg += "\n🟢 <b>準備買入:</b>\n" + "\n".join(buy_msgs) + "\n"
            else:
                msg += "✅ 無買賣異動。\n"
                
            url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
            payload = {
                "chat_id": tg_chat_id,
                "text": msg,
                "parse_mode": "HTML"
            }
            res = requests.post(url, json=payload)
            if res.status_code == 200:
                print("✅ Telegram 異動通知發送成功！")
                # 寫入 Firebase，確保今天所有排程都不再重複發送
                if notify_db:
                    try:
                        notify_db.collection("quant_fund").document("notify_metadata").set({"last_notify_date": today_str})
                        print(f"✅ Firebase 通知記錄已更新: {today_str}")
                    except Exception as e_notify_write:
                        print(f"⚠️ 無法寫入 Firebase 通知記錄: {e_notify_write}")
            else:
                print(f"⚠️ Telegram 發送失敗，狀態碼: {res.status_code}, {res.text}")
        elif already_notified:
            print("💡 今日已發送過異動通知，為避免打擾跳過發送。")
        else:
            print("無買賣異動，跳過發送通知。")
    else:
        print("⚠️ 未設定 TELEGRAM_BOT_TOKEN 或 TELEGRAM_CHAT_ID，跳過發送通知。")
except Exception as e:
    print(f"⚠️ Telegram 執行時發生錯誤: {e}")

# --- Firebase Firestore Upload ---
print("Uploading data to Firebase Firestore...")
try:
    firebase_cert = os.environ.get("FIREBASE_SERVICE_ACCOUNT")
    if firebase_cert:
        cert_dict = json.loads(firebase_cert)
        if not firebase_admin._apps:
            cred = credentials.Certificate(cert_dict)
            firebase_admin.initialize_app(cred)
        
        db = firestore.client()
        safe_data = json.loads(json.dumps(dashboard_data)) # Ensure pure python types
        db.collection("quant_fund").document("dashboard_data").set(safe_data)
        print("✅ Firebase Firestore update successful.")
    else:
        print("⚠️ FIREBASE_SERVICE_ACCOUNT not found. Skipping Firestore update.")
except Exception as e:
    print(f"⚠️ Failed to update Firestore: {e}")

# 產生純前端 HTML (無需注入靜態資料)
current_dir = os.path.dirname(os.path.abspath(__file__))
template_path = os.path.join(current_dir, 'dashboard.html')
index_path = os.path.join(current_dir, 'index.html')

if os.path.exists(template_path):
    import shutil
    shutil.copyfile(template_path, index_path)
    print("Dashboard HTML copied to index.html (Frontend Firebase managed).")

# GitHub 自動同步
if not os.environ.get("GITHUB_ACTIONS"):
    print("📤 [Local Mode] 正在同步全功能數據至雲端...")
    repo_dir = current_dir
    try:
        os.chdir(repo_dir)
        # 建立檔案以避免 git add 找不到報錯
        if not os.path.exists(".last_notify_date"):
            with open(".last_notify_date", "w") as f: pass
        subprocess.run(["git", "add", "index.html", "update_daily.py", ".last_notify_date"], check=True)
        if subprocess.run(["git", "diff", "--staged", "--quiet"]).returncode != 0:
             subprocess.run(["git", "commit", "-m", f"Dashboard Logic Strict Fix: {datetime.now().strftime('%Y-%m-%d %H:%M')}"], check=True)
             subprocess.run(["git", "push", "origin", "main"], check=True)
             print(f"✨ 發布成功！瀏覽網址: https://woody-yiu.github.io/TeraWise-Dashboard/")
        else:
             print("無變更需要提交。")
    except Exception as e:
        print(f"⚠️ 自動發布失敗: {e}")
