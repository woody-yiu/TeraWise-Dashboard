from finlab import login
import pandas as pd
import numpy as np
from finlab import data
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta, timezone
import warnings
import os
import json
import subprocess

warnings.filterwarnings('ignore')

# Enviromment check
token = os.environ.get("FINLAB_TOKEN", "97Y21Yf07Tokqp6rnUxsQKHbc4j+HosTsqE5DNh2oWLA9n+pxaCibJSKUK190ocZ#vip_m")
login(token)

N_DAYS = 5
MIN_LIQ_PCT = 0.6
TOP_GROUPS = 5
TOP_STOCKS = 50           
PORTFOLIO_SIZE = 15       

STOP_LOSS_PCT = 0.10
FIXED_HOLDING_DAYS = 40
COOLING_OFF_DAYS = 5

weights = {"ret": 1.5, "turnover": 1.0, "inst": 1.0, "conc": 1.0}

print("正在抓取並對齊資料...")
close = data.get("price:收盤價")
open_ = data.get("price:開盤價")
volume = data.get("price:成交股數")
benchmark = data.get('taiex_total_index:收盤指數')
benchmark = benchmark[~benchmark.index.duplicated(keep='first')]
benchmark_ma200 = benchmark.rolling(200).mean()

# --- 產業資料 ---
theme_raw = data.get("security_industry_themes") 
cat_raw = data.get("security_categories")

foreign = data.get('institutional_investors_trading_summary:外資自營商買賣超股數')
trust = data.get('institutional_investors_trading_summary:投信買賣超股數')
dealer = data.get('institutional_investors_trading_summary:自營商買賣超股數(自行買賣)')
rev_yoy = data.get("monthly_revenue:去年同月增減(%)")

# 1. 計算用
theme = theme_raw.sort_values("key_date").groupby("stock_id").last()
def parse_category(cat_str):
    try: return cat_str.replace("[", "").replace("]", "").split(",")[0].strip("' \"")
    except: return "Unknown"
theme["main_category"] = theme["category"].apply(parse_category)
group_mapper = theme["main_category"]

# 2. 顯示用
cat_mapper = cat_raw.drop_duplicates("stock_id", keep="last").set_index("stock_id")["category"]

common_cols = close.columns.intersection(volume.columns).intersection(foreign.columns).intersection(trust.columns).intersection(dealer.columns).intersection(rev_yoy.columns).intersection(group_mapper.index)

close = close[common_cols]
open_ = open_[common_cols]
volume = volume[common_cols]
rev_yoy = rev_yoy[common_cols].reindex(close.index, method='ffill')
group_mapper = group_mapper[common_cols]

inst_total = foreign[common_cols] + trust[common_cols] + dealer[common_cols]
inst_buy_yday = inst_total.shift(1).reindex(close.index)
inst_concentration = (inst_total / volume.replace(0, np.nan)).reindex(close.index)
ma200 = close.rolling(200).mean()

print("計算選股訊號...")
ret = close.pct_change(N_DAYS)
turnover = close * volume

group_ret = ret.groupby(group_mapper, axis=1).mean()
group_turnover = turnover.groupby(group_mapper, axis=1).sum()
group_inst = inst_buy_yday.groupby(group_mapper, axis=1).sum()
group_conc = inst_concentration.groupby(group_mapper, axis=1).mean()

group_score = (group_ret.rank(axis=1, pct=True) * weights["ret"] +
               group_turnover.rank(axis=1, pct=True) * weights["turnover"] +
               group_inst.rank(axis=1, pct=True) * weights["inst"] +
               group_conc.rank(axis=1, pct=True) * weights["conc"])

top_groups_daily = group_score.rank(axis=1, ascending=False) <= TOP_GROUPS

selected_stocks_signal = {}
valid_index = close.index.intersection(top_groups_daily.index).intersection(inst_buy_yday.index).intersection(inst_concentration.index)

# 為了加快雲端執行速度，這裡只重算近 500 天 + 所有需要回測的日期，但為了保險起見，我們如果資源允許，還是跑全量
# 觀察: valid_index >= '2011-12-01'
for date in valid_index[valid_index >= '2011-12-01']:
    if date not in top_groups_daily.index: continue
    strong_groups_mask = top_groups_daily.loc[date]
    strong_groups = strong_groups_mask[strong_groups_mask].index.tolist()
    if not strong_groups: continue
    
    # 修正: 使用 intersection 避免 key error
    stocks_in_groups = group_mapper[group_mapper.isin(strong_groups)].index.intersection(close.columns).tolist()
    
    try:
        # 效能優化: 直接取 loc
        df = pd.DataFrame({
            "ret": ret.loc[date, stocks_in_groups],
            "turnover": turnover.loc[date, stocks_in_groups],
            "inst": inst_buy_yday.loc[date, stocks_in_groups],
            "conc": inst_concentration.loc[date, stocks_in_groups],
            "yoy": rev_yoy.loc[date, stocks_in_groups]
        }).dropna()
    except Exception as e: 
        continue
    
    if df.empty: continue
    liq_cut = df["turnover"].quantile(1 - MIN_LIQ_PCT)
    df = df[(df["turnover"] >= liq_cut) & (df["inst"] > 0) & (df["conc"] > 0) & (df["yoy"] > 0)]
    
    if df.empty: continue
    df["score"] = (df["ret"].rank(pct=True) * weights["ret"] +
                  df["turnover"].rank(pct=True) * weights["turnover"] +
                  df["inst"].rank(pct=True) * weights["inst"] +
                  df["conc"].rank(pct=True) * weights["conc"])
    
    # MA200 Filter
    current_ma200 = ma200.loc[date]
    current_close = close.loc[date]
    # 確保 index 對齊
    valid_ma_stocks = df.index.intersection(current_ma200.index).intersection(current_close.index)
    df = df.loc[valid_ma_stocks]
    df = df[current_close[df.index] > current_ma200[df.index]]
    
    selected_stocks_signal[date] = df.sort_values("score", ascending=False).head(TOP_STOCKS).index.tolist()
print("訊號計算完成。")

# --- 終極實戰時序版 ---
print(f"執行回測 (目標持股 {PORTFOLIO_SIZE} 檔，僅限每月 10-15 號進場)...【修正版：次日開盤賣出】")

INITIAL_CAPITAL = 10_000_000
CASH = INITIAL_CAPITAL
PORTFOLIO = []         
PENDING_EXITS = []     
TRADE_LOG = []
NAV_HISTORY = []
last_buy_date = {}

backtest_dates = close.index.intersection(valid_index)
backtest_dates = backtest_dates[backtest_dates >= '2012-01-01']

for i, today in enumerate(backtest_dates):
    yesterday = backtest_dates[i-1] if i > 0 else None
    
    # --- [1] 上午開盤：處理昨晚被標記的「待賣出」股票 ---
    for p in PENDING_EXITS:
        sell_price = open_.at[today, p['stock_id']]
        if pd.isna(sell_price): 
            sell_price = close.at[today, p['stock_id']] 
            
        revenue = sell_price * p['shares']
        fee = revenue * (0.001425 * 0.1 + 0.003) 
        CASH += (revenue - fee)
        
        TRADE_LOG.append({
            'stock_id': p['stock_id'], 'entry_date': p['entry_date'], 'exit_date': today,
            'entry_price': round(p['entry_price'], 2), 'exit_price': round(sell_price, 2),
            'ret': (revenue - fee - p['cost']) / p['cost'], 'exit_reason': p['reason']
        })
    PENDING_EXITS = [] 
    
    # --- [2] 上午開盤：進場邏輯 ---
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
                    
                    # 防呆: 檢查是否有今日開盤價與昨日收盤價 (避免除息或停牌造成錯誤)
                    if pd.isna(open_.at[today, sid]): continue
                    if yesterday not in close.index or pd.isna(close.at[yesterday, sid]): continue
                    if close.at[yesterday, sid] < ma200.at[yesterday, sid]: continue
                    
                    lbd = last_buy_date.get(sid)
                    if lbd and (close.index.get_loc(yesterday) - close.index.get_loc(lbd)) <= COOLING_OFF_DAYS: 
                        continue
                    
                    holdings_val = sum(close.at[today, pp['stock_id']] * pp['shares'] if pd.notna(close.at[today, pp['stock_id']]) else pp['entry_price'] * pp['shares'] for pp in PORTFOLIO)
                    target_value = min((CASH + holdings_val) / PORTFOLIO_SIZE, CASH * 0.98)
                    
                    entry_price = open_.at[today, sid]
                    shares = int(target_value / (entry_price * (1 + 0.001425*0.1)))
                    
                    if shares > 0:
                        cost = entry_price * shares * (1 + 0.001425*0.1)
                        CASH -= cost
                        PORTFOLIO.append({
                            'stock_id': sid, 'entry_date': today, 'entry_price': entry_price, 
                            'shares': shares, 'cost': cost, 'entry_idx': i
                        })
                        last_buy_date[sid] = today
                        slots_to_fill -= 1

    # --- [3] 下午收盤：結算淨值 ---
    current_holdings_value = sum(close.at[today, p['stock_id']] * p['shares'] for p in PORTFOLIO)
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
mdd = df_nav['drawdown'].min()
total_days = (df_nav.index[-1] - df_nav.index[0]).days
annual_ret = (df_nav['nav'].iloc[-1] / INITIAL_CAPITAL)**(365/total_days) - 1
annual_std = df_nav['return'].std() * np.sqrt(252)
sharpe = (annual_ret - 0.02) / annual_std if annual_std != 0 else 0

# --- 7. Dashboard Data Generation ---
print(f"啟動 TeraWise 雲端發布流程 ...")

# 1. Timezone Fix
tz = timezone(timedelta(hours=8))
last_update_str = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

last_data_date = df_nav.index[-1].strftime('%Y-%m-%d')
last_prices = close.iloc[-1]
daily_ret = df_nav['nav'].pct_change().dropna()
ann_ret = (df_nav['nav'].iloc[-1] / df_nav['nav'].iloc[0]) ** (252 / len(df_nav)) - 1
ann_vol = daily_ret.std() * np.sqrt(252)
sharpe = ann_ret / ann_vol if ann_vol != 0 else 0
ann_downside_vol = np.sqrt((daily_ret.clip(upper=0)**2).mean()) * np.sqrt(252)
sortino = ann_ret / ann_downside_vol if ann_downside_vol != 0 else 0
mdd_val = abs(df_nav['drawdown'].min())
calmar = ann_ret / mdd_val if mdd_val != 0 else 0

# 2. Trade Stats
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

# 3. Holdings
curr_holdings_data = []
sector_counter = {}
if PORTFOLIO:
    for p in PORTFOLIO:
        sid = p['stock_id']
        name = theme.loc[sid, 'name'] if sid in theme.index else sid
        cat = cat_mapper.get(sid, "其他")
        sector_counter[cat] = sector_counter.get(cat, 0) + 1
        pnl = (last_prices[sid] / p['entry_price'] - 1)
        curr_holdings_data.append({
            "stock_id": sid, "name": name, "category": cat,
            "entry_date": p['entry_date'].strftime('%Y-%m-%d'),
            "entry_price": round(p['entry_price'], 2), 
            "current_price": round(last_prices[sid], 2),
            "current_date": last_data_date,
            "pnl": round(pnl * 100, 2)
        })
sector_pie = [{"name": k, "value": v} for k, v in sector_counter.items()]

# 4. Signals
recent_signals_data = []
# 檢查是否有訊號，若無則顯示空列表，避免錯誤
if selected_stocks_signal:
    signal_dates = sorted(selected_stocks_signal.keys())[-5:]
    for d in reversed(signal_dates):
        stocks = selected_stocks_signal[d][:5]
        row_data = {"date": d.strftime("%Y-%m-%d"), "stocks": []}
        for i, sid in enumerate(stocks):
            name = theme.loc[sid, 'name'] if sid in theme.index else sid
            cat = cat_mapper.get(sid, "Unknown")
            row_data["stocks"].append(f"{i+1}. {sid} {name} ({cat})")
        recent_signals_data.append(row_data)

# 5. Operations
recent_ops = []
if TRADE_LOG:
    for t in TRADE_LOG:
        recent_ops.append({
            "date": t['exit_date'].strftime('%Y-%m-%d'),
            "action": "賣出",
            "stock_id": t['stock_id'],
            "name": theme.loc[t['stock_id'], 'name'] if t['stock_id'] in theme.index else t['stock_id'],
            "price": round(t['exit_price'], 2),
            "reason": t['exit_reason'],
            "pnl": round(t['ret'] * 100, 2),
            "entry_info": f"({t['entry_date'].strftime('%m/%d')} 以 {round(t['entry_price'], 2)} 買入)"
        })
seen_buys = set()
for t in TRADE_LOG:
    key = (t['stock_id'], t['entry_date'])
    if key not in seen_buys:
        recent_ops.append({
            "date": t['entry_date'].strftime('%Y-%m-%d'),
            "action": "買入",
            "stock_id": t['stock_id'],
            "name": theme.loc[t['stock_id'], 'name'] if t['stock_id'] in theme.index else t['stock_id'],
            "price": round(t['entry_price'], 2),
            "reason": "訊號進場",
            "pnl": "-",
            "entry_info": ""
        })
        seen_buys.add(key)
for p in PORTFOLIO:
    key = (p['stock_id'], p['entry_date'])
    if key not in seen_buys:
        recent_ops.append({
            "date": p['entry_date'].strftime('%Y-%m-%d'),
            "action": "買入",
            "stock_id": p['stock_id'],
            "name": theme.loc[p['stock_id'], 'name'] if p['stock_id'] in theme.index else p['stock_id'],
            "price": round(p['entry_price'], 2),
            "reason": "訊號進場",
            "pnl": "-",
            "entry_info": ""
        })
        seen_buys.add(key)
recent_ops.sort(key=lambda x: x['date'], reverse=True)

# 6. Heatmap
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
for date, group in monthly_groups:
    if len(group) < 2: continue
    yr, mo = str(date.year), date.month
    m_ret = (group['nav'].iloc[-1] / group['nav'].iloc[0] - 1) * 100
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

# 7. History Trades
historical_trades = []
if TRADE_LOG:
    for t in sorted(TRADE_LOG, key=lambda x: x['exit_date'], reverse=True)[:50]:
        historical_trades.append({
            "stock_id": t['stock_id'], "name": theme.loc[t['stock_id'], 'name'] if t['stock_id'] in theme.index else t['stock_id'],
            "category": cat_mapper.get(t['stock_id'], "其他"), "entry_date": t['entry_date'].strftime('%Y-%m-%d'),
            "exit_date": t['exit_date'].strftime('%Y-%m-%d'), "entry_price": round(t['entry_price'], 2),
            "exit_price": round(t['exit_price'], 2), "ret": round(t['ret'], 4), "exit_reason": t['exit_reason']
        })

# Benchmark Alignment
bm_aligned = benchmark.reindex(df_nav.index).ffill()
bm_col = bm_aligned.columns[0]

# --- Final JSON & Output ---
# 關鍵修正: 確保所有欄位都存在，包括 MDD
dashboard_data = {
    "summary": { 
        "last_update": last_update_str, # Fix 1: Timezone
        "sharpe": round(sharpe, 2), 
        "sortino": round(sortino, 2), 
        "calmar": round(calmar, 2), 
        "ann_ret": round(ann_ret * 100, 2),
        "mdd": round(mdd * 100, 2),      # Fix 2: Added MDD
        "downside_risk": round(ann_downside_vol * 100, 2) # Fix 3: Added Downside Risk
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

js_inner = f"var fundData = {json.dumps(dashboard_data, ensure_ascii=False)};"

# Read Template
current_dir = os.path.dirname(os.path.abspath(__file__))
# 嘗試讀取本地 dashboard.html，GitHub runner root 即為 workspace root
template_path = os.path.join(current_dir, 'dashboard.html')
index_path = os.path.join(current_dir, 'index.html')

if not os.path.exists(template_path):
    print(f"Template not found at {template_path}, listing dir:")
    print(os.listdir(current_dir))

with open(template_path, 'r', encoding='utf-8') as f:
    full_html = f.read()

final_html = full_html.replace('<script src="data.js"></script>', f'<script>{js_inner}</script>')

with open(index_path, 'w', encoding='utf-8') as f:
    f.write(final_html)

print("update_daily.py completed successfully with MDD and Timezone fix.")
