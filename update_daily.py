from finlab import login
import pandas as pd
import numpy as np
from finlab import data
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
import json
import subprocess

warnings.filterwarnings('ignore')

# login('97Y21Yf07Tokqp6rnUxsQKHbc4j+HosTsqE5DNh2oWLA9n+pxaCibJSKUK190ocZ#vip_m')
# 為了 GitHub Actions 環境，改用環境變數讀取 Token，若無則使用預設值 (保留給本地測試用)
token = os.environ.get("FINLAB_TOKEN", "97Y21Yf07Tokqp6rnUxsQKHbc4j+HosTsqE5DNh2oWLA9n+pxaCibJSKUK190ocZ#vip_m")
login(token)

N_DAYS = 5
MIN_LIQ_PCT = 0.6
TOP_GROUPS = 5
TOP_STOCKS = 20           
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

# --- 產業資料：同時抓取題材標籤(計算用)與官方分類(顯示用) ---
theme_raw = data.get("security_industry_themes") 
cat_raw = data.get("security_categories")

foreign = data.get('institutional_investors_trading_summary:外資自營商買賣超股數')
trust = data.get('institutional_investors_trading_summary:投信買賣超股數')
dealer = data.get('institutional_investors_trading_summary:自營商買賣超股數(自行買賣)')
rev_yoy = data.get("monthly_revenue:去年同月增減(%)")

# 1. 計算用：維持細分標籤邏輯 (不影響原本選股結果)
theme = theme_raw.sort_values("key_date").groupby("stock_id").last()
def parse_category(cat_str):
    try: return cat_str.replace("[", "").replace("]", "").split(",")[0].strip("' \"")
    except: return "Unknown"
theme["main_category"] = theme["category"].apply(parse_category)
group_mapper = theme["main_category"]

# 2. 顯示用：正確的官方產業名稱 (讓廣達回歸電腦週邊)
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

for date in valid_index[valid_index >= '2011-12-01']:
    if date not in top_groups_daily.index: continue
    strong_groups_mask = top_groups_daily.loc[date]
    strong_groups = strong_groups_mask[strong_groups_mask].index.tolist()
    if not strong_groups: continue
    
    stocks_in_groups = group_mapper[group_mapper.isin(strong_groups)].index.tolist()
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
    
    # 🚀 在這裡加入個股 MA200 濾網
    # 只保留當天收盤價大於 200 日均線的股票
    current_ma200 = ma200.loc[date]
    current_close = close.loc[date]
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

for i, today in enumerate(backtest_dates):
    yesterday = backtest_dates[i-1] if i > 0 else None
    
    # --- [1] 上午開盤：處理昨晚被標記的「待賣出」股票 ---
    for p in PENDING_EXITS:
        sell_price = open_.at[today, p['stock_id']]
        if pd.isna(sell_price): 
            sell_price = close.at[today, p['stock_id']] # 若無開盤價則以今日收盤代替
            
        revenue = sell_price * p['shares']
        fee = revenue * (0.001425 * 0.1 + 0.003) 
        CASH += (revenue - fee)
        
        TRADE_LOG.append({
            'stock_id': p['stock_id'], 'entry_date': p['entry_date'], 'exit_date': today,
            'entry_price': round(p['entry_price'], 2), 'exit_price': round(sell_price, 2),
            'ret': (revenue - fee - p['cost']) / p['cost'], 'exit_reason': p['reason']
        })
    PENDING_EXITS = [] # 執行完畢，清空待賣清單，此時空位才真正釋放
    
    # --- [2] 上午開盤：進場邏輯 (檢查空位併補貨) ---
    if yesterday is not None:
        market_pass = True
        bm_yesterday = benchmark.at[yesterday, benchmark.columns[0]] if yesterday in benchmark.index else np.nan
        bm_ma_yesterday = benchmark_ma200.at[yesterday, benchmark_ma200.columns[0]] if yesterday in benchmark_ma200.index else np.nan
        if pd.notna(bm_yesterday) and pd.notna(bm_ma_yesterday) and bm_yesterday < bm_ma_yesterday: 
            market_pass = False
        
        is_entry_window = 10 <= today.day <= 15
                
        if market_pass and is_entry_window:
            # 此時的 PORTFOLIO 已不包含昨晚標記的股票
            slots_to_fill = PORTFOLIO_SIZE - len(PORTFOLIO)
            if slots_to_fill > 0:
                signals = selected_stocks_signal.get(yesterday, [])
                for sid in signals:
                    if slots_to_fill <= 0: break
                    if any(p['stock_id'] == sid for p in PORTFOLIO): continue
                    if pd.isna(open_.at[today, sid]): continue
                    if close.at[yesterday, sid] < ma200.at[yesterday, sid]: continue
                    
                    lbd = last_buy_date.get(sid)
                    if lbd and (close.index.get_loc(yesterday) - close.index.get_loc(lbd)) <= COOLING_OFF_DAYS: 
                        continue
                    
                    # 計算淨值基準 (今日開盤前的淨值)
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

    # --- [3] 下午收盤：結算淨值與判定明日賣出清單 ---
    # 今日終了淨值
    current_holdings_value = sum(close.at[today, p['stock_id']] * p['shares'] for p in PORTFOLIO)
    current_nav = CASH + current_holdings_value
    
    # 檢查是否有股票觸發停損或到期 (今日收盤判定，明日開盤執行)
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
            # 移出持倉，改放入待賣清單
            p['reason'] = exit_reason
            PENDING_EXITS.append(p)
        else:
            new_active_portfolio.append(p)
    
    PORTFOLIO = new_active_portfolio # 更新持倉清單

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
metrics = {
    "累積總報酬": f"{((df_nav['nav'].iloc[-1]/INITIAL_CAPITAL)-1)*100:.2f}%",
    "年化報酬率": f"{annual_ret*100:.2f}%",
    "最大回撤 (MDD)": f"{mdd*100:.2f}%",
    "夏普比率 (Sharpe)": f"{sharpe:.2f}",
    "平均持股數": f"{df_nav['holdings_count'].mean():.1f} 檔"
}
# display(pd.DataFrame(list(metrics.items()), columns=['指標', '數值']))

# --- 終極自動化：數據分析 + 訊號補回 + 雲端發布 (全功能完整版) ---
# import json
# import os
# import subprocess
# import pandas as pd
# import numpy as np
# from datetime import datetime
print(f"🚀 啟動 TeraWise 雲端發布流程 [{datetime.now().strftime('%Y-%m-%d %H:%M')}]...")
# 1. 基礎指標計算 (Sharpe, MDD, CAGR)
last_data_date = df_nav.index[-1].strftime('%Y-%m-%d')
last_prices = close.iloc[-1]
daily_ret = df_nav['nav'].pct_change().dropna()
ann_ret = (df_nav['nav'].iloc[-1] / df_nav['nav'].iloc[0]) ** (252 / len(df_nav)) - 1
ann_vol = daily_ret.std() * np.sqrt(252)
sharpe = ann_ret / ann_vol if ann_vol != 0 else 0
downside_ret = daily_ret[daily_ret < 0]
ann_downside_vol = downside_ret.std() * np.sqrt(252)
sortino = ann_ret / ann_downside_vol if ann_downside_vol != 0 else 0
mdd_val = abs(df_nav['drawdown'].min())
calmar = ann_ret / mdd_val if mdd_val != 0 else 0
# 2. 交易勝報比統計
trade_stats = {"win_rate": 0, "avg_win": 0, "avg_loss": 0, "profit_factor": 0, "total_trades": 0}
if TRADE_LOG:
    rets = [t['ret'] for t in TRADE_LOG]
    wins = [r for r in rets if r > 0]; losses = [r for r in rets if r <= 0]
    trade_stats = {
        "win_rate": round(len(wins)/len(rets)*100, 2), "avg_win": round(np.mean(wins)*100, 2) if wins else 0,
        "avg_loss": round(np.mean(losses)*100, 2) if losses else 0, "profit_factor": round(sum(wins)/abs(sum(losses)), 2) if losses and sum(losses)!=0 else 0,
        "total_trades": len(rets)
    }
# 3. 處理持倉細節
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
# 4. 🚀 補回最近 5 日選股訊號 (加上排名排名)
recent_signals_data = []
if 'selected_stocks_signal' in locals() and selected_stocks_signal:
    signal_dates = sorted(selected_stocks_signal.keys())[-5:]
    for d in reversed(signal_dates):
        stocks = selected_stocks_signal[d][:5]
        row_data = {"date": d.strftime("%Y-%m-%d"), "stocks": []}
        for i, sid in enumerate(stocks):
            name = theme.loc[sid, 'name'] if sid in theme.index else sid
            cat = cat_mapper.get(sid, "Unknown")
            row_data["stocks"].append(f"{i+1}. {sid} {name} ({cat})")
        recent_signals_data.append(row_data)
# 5. 🚀 產生「近期基金操作日誌」(Activity Log)
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
    # 邏輯: 確保持倉的股票買入紀錄也存在
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
for date, group in monthly_groups:
    if len(group) < 2: continue
    yr, mo = str(date.year), date.month
    m_ret = (group['nav'].iloc[-1] / group['nav'].iloc[0] - 1) * 100
    test_day = group.index[-1]
    m_stock_rets = (close.loc[test_day] / close.loc[group.index[0]] - 1)
    # Check for empty or non-matching index
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
        historical_trades.append({
            "stock_id": t['stock_id'], "name": theme.loc[t['stock_id'], 'name'] if t['stock_id'] in theme.index else t['stock_id'],
            "category": cat_mapper.get(t['stock_id'], "其他"), "entry_date": t['entry_date'].strftime('%Y-%m-%d'),
            "exit_date": t['exit_date'].strftime('%Y-%m-%d'), "entry_price": round(t['entry_price'], 2),
            "exit_price": round(t['exit_price'], 2), "ret": round(t['ret'], 4), "exit_reason": t['exit_reason']
        })
# 8. 建構封包並發布
# --- 關鍵修復：對齊大盤資料並填充 ---
bm_aligned = benchmark.reindex(df_nav.index).ffill()
bm_col = bm_aligned.columns[0]
dashboard_data = {
    "summary": { "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "sharpe": round(sharpe, 2), "sortino": round(sortino, 2), "calmar": round(calmar, 2), "ann_ret": round(ann_ret * 100, 2) },
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

# 修改路徑以適應 GitHub Runner 環境
template_path = 'dashboard.html' # r'c:\Users\teraw_rp58jwl\OneDrive\桌面\量化選股策略\dashboard.html'
# 確保如果本地測試也能跑，可以使用相對路徑或檢查檔案是否存在
if not os.path.exists(template_path):
    # 若本地執行，嘗試絕對路徑 (這裡僅作範容錯，主要給遠端用)
    pass 

with open(template_path, 'r', encoding='utf-8') as f:
    full_html = f.read()
final_html = full_html.replace('<script src="data.js"></script>', f'<script>{js_inner}</script>')
index_path = 'index.html' # r'c:\Users\teraw_rp58jwl\OneDrive\桌面\量化選股策略\index.html'
with open(index_path, 'w', encoding='utf-8') as f:
    f.write(final_html)

# GitHub 自動同步
# repo_dir = r'c:\Users\teraw_rp58jwl\OneDrive\桌面\量化選股策略'
# print("📤 正在同步全功能數據至雲端...")
# try:
#     os.chdir(repo_dir)
#     subprocess.run(["git", "add", "index.html"], check=True, capture_output=True)
#     subprocess.run(["git", "commit", "-m", f"Dashboard Benchmark Fix: {datetime.now().strftime('%Y-%m-%d %H:%M')}"], check=True, capture_output=True)
#     subprocess.run(["git", "push", "origin", "main"], check=True, capture_output=True)
#     print(f"✨ 發布成功！瀏覽網址: https://woody-yiu.github.io/TeraWise-Dashboard/")
# except Exception as e:
#     print(f"⚠️ 自動發布失敗: {e}")
# (以上 Git 操作已由 GitHub Actions 統一託管)
