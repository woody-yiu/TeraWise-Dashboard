
import json
import os

def patch_notebook(filepath, updates):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            for old_text, new_text in updates.items():
                if old_text in source:
                    source = source.replace(old_text, new_text)
                    modified = True
            cell['source'] = [line + '\n' for line in source.split('\n')]
            if cell['source'][-1] == '\n':
                cell['source'].pop()
                
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print(f"Patched {filepath}")
    else:
        print(f"No changes made to {filepath}")

# Updates for 找強勢族群_YOY.ipynb
yoy_updates = {
    'theme_raw = data.get("security_industry_themes")': 'theme_raw = data.get("security_industry_themes")\n    cat_raw = data.get("security_categories")',
    'group_mapper = theme["main_category"]': 'group_mapper = theme["main_category"]\n    cat_mapper = cat_raw.drop_duplicates("stock_id", keep="last").set_index("stock_id")["category"]',
    'category = row[\'main_category\'] if \'main_category\' in row else \'\'': 'category = cat_mapper.get(stock_id, "Unknown")'
}

# Updates for 基金選股_15檔.ipynb
fund_updates = {
    'theme_raw = data.get("security_industry_themes")': 'theme_raw = data.get("security_industry_themes")\n                cat_raw = data.get("security_categories")',
    'group_mapper = theme["main_category"]': 'group_mapper = theme["main_category"]\n                cat_mapper = cat_raw.drop_duplicates("stock_id", keep="last").set_index("stock_id")["category"]',
    'return theme.loc[stock_id, \'main_category\'] if stock_id in theme.index else ""': 'return cat_mapper.get(stock_id, "Unknown")',
    'cat_data = theme.loc[sid, \'category\'] if sid in theme.index and \'category\' in theme.columns else ""\n        \n        main_cat = ""\n        if isinstance(cat_data, (list, tuple)) and len(cat_data) > 0:\n            main_cat = str(cat_data[0])\n        elif isinstance(cat_data, str):\n            # 如果是字串形式的列表 "( \'A\', \'B\' )"，只取第一個引號內的字\n            main_cat = cat_data.replace("(\'", "").replace("\')", "").split("\', \'")[0].split(":")[0]\n            # 去除多餘字符\n            main_cat = main_cat.strip("()[]\'\\" ")': 'main_cat = cat_mapper.get(sid, "Unknown")'
}

patch_notebook(r'c:\Users\teraw_rp58jwl\OneDrive\桌面\量化選股策略\找強勢族群_YOY.ipynb', yoy_updates)
patch_notebook(r'c:\Users\teraw_rp58jwl\OneDrive\桌面\量化選股策略\基金選股_15檔.ipynb', fund_updates)
