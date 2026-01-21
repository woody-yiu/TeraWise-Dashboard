
import json
import os

notebook_path = r'c:\Users\teraw_rp58jwl\OneDrive\桌面\量化選股策略\基金選股_15檔.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# --- 1. Patch Data Fetching ---
for cell in nb['cells']:
    source = "".join(cell['source'])
    if 'shared_index = close.index' not in source and 'price:收盤價' in source:
        # Insert date alignment after getting benchmark
        old_pattern = 'benchmark = data.get(\'taiex_total_index:收盤指數\')'
        new_logic = 'benchmark = data.get(\'taiex_total_index:收盤指數\')\n    \n    # 【關鍵修正 1】對齊指數 (確保大盤與個股日期一致，避免日期不齊導致報錯)\n    shared_index = close.index.intersection(benchmark.index)\n    close = close.loc[shared_index]\n    open_p = open_p.loc[shared_index]\n    volume = volume.loc[shared_index]\n    benchmark = benchmark.loc[shared_index]'
        if old_pattern in source:
            source = source.replace(old_pattern, new_logic)
            cell['source'] = [line + '\n' for line in source.split('\n')]
            if cell['source'][-1] == '\n': cell['source'].pop()

# --- 2. Patch Signal Generation ---
for cell in nb['cells']:
    source = "".join(cell['source'])
    if 'stocks_in_groups = group_mapper' in source and 'intersection(close.columns)' not in source:
        old_pattern = 'stocks_in_groups = group_mapper[group_mapper.isin(strong_groups)].index.tolist()'
        new_logic = '    # 【關鍵修正 2】確保只取「有價格數據」的成分股 (intersection)\n    stocks_in_groups = group_mapper[group_mapper.isin(strong_groups)].index.intersection(close.columns).tolist()'
        if old_pattern in source:
            source = source.replace(old_pattern, new_logic)
            cell['source'] = [line + '\n' for line in source.split('\n')]
            if cell['source'][-1] == '\n': cell['source'].pop()

# --- 3. Patch Backtest ---
for cell in nb['cells']:
    source = "".join(cell['source'])
    if 'nav_history.append' in source and 'bm_val' not in source:
        old_pattern = "nav_history.append({\n            'date': today,\n            'nav': current_nav,\n            'benchmark': benchmark.at[today, benchmark.columns[0]],\n        })"
        new_logic = "        bm_val = benchmark.at[today, benchmark.columns[0]] if today in benchmark.index else (nav_history[-1]['benchmark'] if nav_history else 0)\n        nav_history.append({\n            'date': today,\n            'nav': current_nav,\n            'benchmark': bm_val\n        })"
        if old_pattern in source:
            source = source.replace(old_pattern, new_logic)
            cell['source'] = [line + '\n' for line in source.split('\n')]
            if cell['source'][-1] == '\n': cell['source'].pop()

# --- 4. Replace Final Automation Cell ---
automation_code = """import json
import os
import time
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from finlab import data, login

# --- 設定區域 ---
FINLAB_TOKEN = "97Y21Yf07Tokqp6rnUxsQKHbc4j+HosTsqE5DNh2oWLA9n+pxaCibJSKUK190ocZ#vip_m"
BASE_DIR = r"c:\\Users\\teraw_rp58jwl\\OneDrive\\桌面\\量化選股策略"
DASHBOARD_PATH = os.path.join(BASE_DIR, "dashboard.html")
INDEX_PATH = os.path.join(BASE_DIR, "index.html")

def run_daily_update_task():
    print(f"\\n🔔 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 啟動 TeraWise 全自動更新流程...")
    
    try:
        # 1. 登入與資料抓取
        login(FINLAB_TOKEN)
        print("📡 正在抓取最新市場數據...")
        
        close = data.get("price:收盤價")
        open_p = data.get("price:開盤價")
        volume = data.get("price:成交股數")
        benchmark = data.get('taiex_total_index:收盤指數')
        
        # 額外因子
        foreign = data.get('institutional_investors_trading_summary:外資自營商買賣超股數')
        trust = data.get('institutional_investors_trading_summary:投信買賣超股數')
        dealer = data.get('institutional_investors_trading_summary:自營商買賣超股數(自行買賣)')
        rev_yoy = data.get("monthly_revenue:去年同月增減(%)")
        theme_raw = data.get("security_industry_themes")
        cat_raw = data.get("security_categories")

        # --- 數據清理與對齊 ---
        shared_dates = close.index.intersection(benchmark.index).intersection(volume.index)
        close = close.loc[shared_dates]; open_p = open_p.loc[shared_dates]; volume = volume.loc[shared_dates]; benchmark = benchmark.loc[shared_dates]
        
        cat_mapper = cat_raw.drop_duplicates("stock_id", keep="last").set_index("stock_id")["category"].to_dict()
        def get_main_cat(sid): return cat_mapper.get(sid, "其他")
        
        common = close.columns.intersection(volume.columns).intersection(foreign.columns).intersection(rev_yoy.columns)
        close = close[common]; volume = volume[common]; rev_yoy = rev_yoy[common].reindex(close.index, method='ffill')
        
        # 2. 策略計算
        print("🧠 計算選股訊號...")
        ret_5d = close.pct_change(5)
        turnover = close * volume
        inst_total = (foreign[common] + trust[common] + dealer.reindex(columns=common).fillna(0))
        inst_concentration = (inst_total / volume.replace(0, np.nan)).reindex(close.index)
        ma200 = close.rolling(200).mean()
        benchmark_ma200 = benchmark.rolling(200).mean()
        
        group_series = pd.Series(cat_mapper)
        g_ret = ret_5d.groupby(group_series, axis=1).mean()
        g_vol = turnover.groupby(group_series, axis=1).sum()
        g_inst = inst_total.shift(1).groupby(group_series, axis=1).sum()
        g_conc = inst_concentration.groupby(group_series, axis=1).mean()
        
        g_score = (g_ret.rank(axis=1, pct=True) * 1.5 + g_vol.rank(axis=1, pct=True) * 1.0 + 
                   g_inst.rank(axis=1, pct=True) * 1.0 + g_conc.rank(axis=1, pct=True) * 1.0)
        
        top_groups_mask = g_score.rank(axis=1, ascending=False) <= 5
        signals = {}
        valid_dates = close.index[200:]
        
        for date in valid_dates[-10:]:
            strong = top_groups_mask.loc[date][top_groups_mask.loc[date]].index.tolist()
            stocks = group_series[group_series.isin(strong)].index.intersection(close.columns)
            if stocks.empty: continue
            df_s = pd.DataFrame({"ret": ret_5d.loc[date, stocks], "turnover": turnover.loc[date, stocks], 
                                 "inst": inst_total.loc[date, stocks], "conc": inst_concentration.loc[date, stocks],
                                 "yoy": rev_yoy.loc[date, stocks]}).dropna()
            df_s = df_s[(df_s["inst"] > 0) & (df_s["yoy"] > 0)]
            if df_s.empty: continue
            df_s["score"] = df_s.rank(pct=True).sum(axis=1)
            signals[date] = df_s.sort_values("score", ascending=False).head(20).index.tolist()

        # 3. 執行回測
        print("🔄 執行歷史回測更新...")
        CASH = 10_000_000; PORTFOLIO = []; TRADE_LOG = []; NAV_HISTORY = []
        PORTFOLIO_SIZE = 15; FIXED_HOLDINGS = 40; STOP_LOSS = 0.10
        
        for i, today in enumerate(valid_dates):
            yesterday = valid_dates[i-1] if i > 0 else None
            holdings_val = sum( (close.at[today, p['stock_id']] if pd.notna(close.at[today, p['stock_id']]) else p['entry_price']) * p['shares'] for p in PORTFOLIO )
            current_nav = CASH + holdings_val
            
            new_p = []
            for p in PORTFOLIO:
                cp = close.at[today, p['stock_id']]
                reason = None
                if pd.notna(cp):
                    if cp < p['entry_price'] * (1 - STOP_LOSS): reason = "Stop Loss"
                    elif (i - p['entry_idx']) >= FIXED_HOLDINGS: reason = "Time Exit"
                if reason:
                    val = cp * p['shares']; fee = val * 0.0044; CASH += (val - fee)
                    TRADE_LOG.append({'stock_id': p['stock_id'], 'entry_date': p['entry_date'], 'exit_date': today, 'entry_price': p['entry_price'], 'exit_price': cp, 'ret': (val-fee-p['cost'])/p['cost'], 'exit_reason': reason})
                else: new_p.append(p)
            PORTFOLIO = new_p
            
            if yesterday is not None and 10 <= today.day <= 15 and len(PORTFOLIO) < PORTFOLIO_SIZE:
                bm_p = benchmark.at[yesterday, benchmark.columns[0]]
                bm_ma = benchmark_ma200.at[yesterday, benchmark_ma200.columns[0]]
                if bm_p > bm_ma:
                    for sid in signals.get(yesterday, []):
                        if len(PORTFOLIO) >= PORTFOLIO_SIZE: break
                        if any(x['stock_id'] == sid for x in PORTFOLIO): continue
                        if sid not in close.columns or pd.isna(close.at[yesterday, sid]): continue
                        if close.at[yesterday, sid] < ma200.at[yesterday, sid]: continue
                        
                        target = min(current_nav/PORTFOLIO_SIZE, CASH * 0.95)
                        ep = open_p.at[today, sid]
                        shares = int(target / (ep * 1.001425))
                        if shares > 0:
                            cost = ep * shares * 1.001425; CASH -= cost
                            PORTFOLIO.append({'stock_id': sid, 'entry_date': today, 'entry_price': ep, 'shares': shares, 'cost': cost, 'entry_idx': i})
            
            bm_val = benchmark.at[today, benchmark.columns[0]] if today in benchmark.index else (0 if not NAV_HISTORY else NAV_HISTORY[-1]['benchmark'])
            NAV_HISTORY.append({'date': today, 'nav': current_nav, 'benchmark': bm_val})

        # 4. 指標產生
        print("📦 產生儀表板數據...")
        theme_names = data.get("security_industry_themes").sort_values("key_date").groupby("stock_id").last()
        df_nav = pd.DataFrame(NAV_HISTORY).set_index('date')
        df_nav['drawdown'] = df_nav['nav'] / df_nav['nav'].cummax() - 1
        ann_ret = (df_nav['nav'].iloc[-1] / df_nav['nav'].iloc[0]) ** (252/len(df_nav)) - 1
        daily_rets = df_nav['nav'].pct_change().dropna()
        sharpe = (ann_ret / (daily_rets.std() * np.sqrt(252))) if daily_rets.std() != 0 else 0
        down_rets = daily_rets[daily_rets < 0]
        sortino = (ann_ret / (down_rets.std() * np.sqrt(252))) if not down_rets.empty else 0
        mdd = abs(df_nav['drawdown'].min())
        calmar = ann_ret / mdd if mdd > 0 else 0
        
        last_prices = close.iloc[-1]
        dashboard_data = {
            "summary": { "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "sharpe": round(sharpe, 2), "sortino": round(sortino, 2), "calmar": round(calmar, 2), "ann_ret": round(ann_ret * 100, 2) },
            "trade_stats": { "win_rate": round(sum(1 for t in TRADE_LOG if t['ret']>0)/len(TRADE_LOG)*100, 2) if TRADE_LOG else 0, "avg_win": round(np.mean([t['ret'] for t in TRADE_LOG if t['ret']>0])*100, 2) if any(t['ret']>0 for t in TRADE_LOG) else 0, "avg_loss": round(np.mean([t['ret'] for t in TRADE_LOG if t['ret']<=0])*100, 2) if any(t['ret']<=0 for t in TRADE_LOG) else 0, "profit_factor": round(sum(t['ret'] for t in TRADE_LOG if t['ret']>0)/abs(sum(t['ret'] for t in TRADE_LOG if t['ret']<=0)), 2) if any(t['ret']<=0 for t in TRADE_LOG) else 0, "total_trades": len(TRADE_LOG) },
            "current_holdings": [{ "stock_id": p['stock_id'], "name": theme_names.loc[p['stock_id'], 'name'] if p['stock_id'] in theme_names.index else p['stock_id'], "category": get_main_cat(p['stock_id']), "entry_date": p['entry_date'].strftime('%Y-%m-%d'), "entry_price": round(p['entry_price'], 2), "current_price": round(last_prices[p['stock_id']], 2), "current_date": df_nav.index[-1].strftime('%Y-%m-%d'), "pnl": round((last_prices[p['stock_id']]/p['entry_price']-1)*100, 2) } for p in PORTFOLIO],
            "recent_signals": [{ "date": d.strftime("%Y-%m-%d"), "stocks": [f"{s} {theme_names.loc[s, 'name'] if s in theme_names.index else ''} ({get_main_cat(s)})" for s in sids[:5]] } for d, sids in sorted(signals.items())[-5:][::-1]],
            "trades": [{ "stock_id": t['stock_id'], "name": theme_names.loc[t['stock_id'], 'name'] if t['stock_id'] in theme_names.index else t['stock_id'], "category": get_main_cat(t['stock_id']), "entry_date": t['entry_date'].strftime('%Y-%m-%d'), "exit_date": t['exit_date'].strftime('%Y-%m-%d'), "entry_price": round(t['entry_price'], 2), "exit_price": round(t['exit_price'], 2), "ret": round(t['ret'], 4), "exit_reason": t['exit_reason'] } for t in TRADE_LOG[-50:][::-1]],
            "sectors": [{ "name": k, "value": int(v) } for k, v in pd.Series([get_main_cat(p['stock_id']) for p in PORTFOLIO]).value_counts().items()],
            "heatmap": {}, 
            "history": [{ "date": d.strftime("%Y-%m-%d"), "nav": round(v, 2), "benchmark": round(b, 2), "mdd": round(m*100, 2) } for d, v, b, m in zip(df_nav.index, df_nav['nav'], df_nav['benchmark'], df_nav['drawdown'])]
        }
        
        js_content = f"var fundData = {json.dumps(dashboard_data, ensure_ascii=False)};"
        with open(DASHBOARD_PATH, 'r', encoding='utf-8') as f: html_tpl = f.read()
        with open(INDEX_PATH, 'w', encoding='utf-8') as f: f.write(html_tpl.replace('<script src="data.js"></script>', f'<script>{js_content}</script>'))

        # 6. Git Push
        print("📤 推送數據至 GitHub...")
        os.chdir(BASE_DIR)
        subprocess.run(["git", "add", "index.html"], check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", f"Auto Update {datetime.now().strftime('%m/%d %H:%M')}"], check=True, capture_output=True)
        subprocess.run(["git", "push", "origin", "main"], check=True, capture_output=True)
        print("✨ 更新完成！")

    except Exception as e:
        print(f"❌ 更新失敗: {e}")

# --- 主循環 ---
while True:
    run_daily_update_task()
    now = datetime.now()
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    sleep_secs = (next_hour - now).total_seconds()
    print(f"💤 [{now.strftime('%H:%M:%S')}] 休息中... 下次更新: {next_hour.strftime('%H:%M:%S')}")
    time.sleep(sleep_secs)
"""

# Replace the last cell (assumed to be automation or create if not exists)
# Find the cell starting with '# --- 終極自動化' or similar
replaced_final = False
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and ('終極自動化' in "".join(cell['source']) or '啟動 TeraWise' in "".join(cell['source'])):
        nb['cells'][i]['source'] = [line + '\n' for line in automation_code.split('\n')]
        if nb['cells'][i]['source'][-1] == '\n': nb['cells'][i]['source'].pop()
        replaced_final = True
        break

if not replaced_final:
    # Append if not found
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\n' for line in automation_code.split('\n')]
    }
    if new_cell['source'][-1] == '\n': new_cell['source'].pop()
    nb['cells'].append(new_cell)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Notebook patched successfully!")
