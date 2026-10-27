"""Private New Noble positions for the website's daily Telegram.

Preview: python new_noble_notification.py
Sync:    python new_noble_notification.py --sync --credentials <service-account.json>
Sync reads the workbook; it never changes trades or sends Telegram messages.
The website supplies its existing ranking, market filters and Taiwan calendar.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

TAIPEI = timezone(timedelta(hours=8))
PRIVATE_COLLECTION = "private_client_notifications"
PRIVATE_DOCUMENT = "new_noble"
DEFAULT_WORKBOOK = (
    Path.home() / "OneDrive" / "桌面" / "在家版＿量化選股"
    / "客戶交易紀錄" / "New Noble客戶" / "TradeLog_New_Noble.xlsx"
)


def _sid(value):
    if isinstance(value, (int, float)):
        if not math.isfinite(value) or int(value) != value:
            raise ValueError("股票代號無效")
        return str(int(value))
    value = str(value).strip()
    if not value or value.lower() == "nan":
        raise ValueError("股票代號無效")
    return value


def read_positions(workbook=DEFAULT_WORKBOOK, *, now=None):
    """Replay actual shares; partial sales proportionally reduce stop-basis lots."""
    from openpyxl import load_workbook

    now = now or datetime.now(TAIPEI)
    workbook = Path(workbook)
    before = hashlib.sha256(workbook.read_bytes()).hexdigest()
    book = load_workbook(workbook, read_only=True, data_only=True)
    try:
        rows = iter(book.worksheets[0].values)
        headers = next(rows)
        required = {"日期", "股票代號", "買賣", "成交價", "股數"}
        if not required.issubset(headers):
            raise ValueError("交易表缺少必要欄位")
        trades = []
        for number, row in enumerate(rows, 2):
            if not any(v is not None for v in row):
                continue
            trade = dict(zip(headers, row))
            if any(word in str(trade.get("備註", "")) for word in ("股票股利", "配股")):
                continue  # Same raw-share ledger convention as the client tracker.
            if any(trade.get(k) is None for k in required):
                raise ValueError(f"第 {number} 列交易資料不完整")
            trade["日期"] = pd.Timestamp(trade["日期"]).normalize()
            if trade["日期"] > pd.Timestamp(now.date()):
                raise ValueError(f"第 {number} 列含未來交易")
            trades.append(trade)
    finally:
        book.close()
    if not trades:
        raise ValueError("交易表沒有可用交易，不能當作空倉")
    cash = 11_000_000.0  # Existing New Noble tracker initial capital.
    positions, last_buys = {}, {}
    for trade in sorted(trades, key=lambda t: t["日期"]):
        sid = _sid(trade["股票代號"])
        shares, price = float(trade["股數"]), float(trade["成交價"])
        fee = trade.get("手續費及稅")
        fee = 0.0 if fee is None else float(fee)
        if not all(math.isfinite(v) for v in (shares, price, fee)):
            raise ValueError(f"{sid} 交易數值無效")
        if shares <= 0 or int(shares) != shares or price <= 0 or fee < 0:
            raise ValueError(f"{sid} 股數、價格或費用無效")
        shares = int(shares)
        date = trade["日期"].strftime("%Y-%m-%d")
        action = str(trade["買賣"]).strip()
        if action == "買":
            p = positions.setdefault(sid, {"shares": 0, "entry_date": date, "lots": []})
            p["shares"] += shares
            p["lots"].append({"date": date, "shares": float(shares)})
            cash -= price * shares + fee
            last_buys[sid] = date
        elif action == "賣":
            if sid not in positions or shares > positions[sid]["shares"]:
                raise ValueError(f"{date} {sid} 賣出超過實際持股")
            p = positions[sid]
            ratio = (p["shares"] - shares) / p["shares"]
            for lot in p["lots"]:
                lot["shares"] *= ratio
            p["shares"] -= shares
            if not p["shares"]:
                del positions[sid]
            cash += price * shares - fee
        else:
            raise ValueError(f"{sid} 僅支援買／賣交易")
    if before != hashlib.sha256(workbook.read_bytes()).hexdigest():
        raise ValueError("讀取期間 Excel 已變更，請重新同步")
    return {
        "schema_version": 1, "client": "New Noble",
        "checked_at": now.isoformat(), "source_sha256": before,
        "latest_trade_date": max(t["日期"] for t in trades).strftime("%Y-%m-%d"),
        "cash": cash, "positions": positions, "last_buy_dates": last_buys,
    }


def get_database(credentials_path=None):
    import firebase_admin
    from firebase_admin import credentials, firestore

    if not firebase_admin._apps:
        raw = os.environ.get("FIREBASE_SERVICE_ACCOUNT")
        path = credentials_path or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not raw and not path:
            candidates = list((Path(__file__).resolve().parent / "TeraWisePrivate").glob("*adminsdk*.json"))
            if len(candidates) == 1:
                path = candidates[0]
        if not raw and not path:
            raise ValueError("缺少 Firebase 管理員憑證；請用 --credentials 指定 JSON 路徑")
        cert = json.loads(raw) if raw else str(Path(path).expanduser().resolve())
        firebase_admin.initialize_app(credentials.Certificate(cert))
    return firestore.client()


def sync_positions(snapshot, database):
    """Refuse upload if unauthenticated Firestore access is possible or uncertain."""
    import requests

    endpoint = (
        f"https://firestore.googleapis.com/v1/projects/{database.project}"
        f"/databases/(default)/documents/{PRIVATE_COLLECTION}/{PRIVATE_DOCUMENT}"
    )
    response = requests.get(endpoint, timeout=20)
    error = response.json().get("error", {})
    if response.status_code != 403 or error.get("status") != "PERMISSION_DENIED":
        raise ValueError("尚未確認 Firebase 私有文件拒絕未登入讀取，停止同步；請檢查安全規則")
    # Encrypt for every active key of the same service account. A newly downloaded
    # local JSON and the existing GitHub JSON can use different active key IDs.
    import firebase_admin
    from urllib.parse import quote

    email = firebase_admin.get_app().credential.get_credential().service_account_email
    cert_response = requests.get(
        "https://www.googleapis.com/service_accounts/v1/metadata/x509/" + quote(email, safe=""),
        timeout=20,
    )
    cert_response.raise_for_status()
    envelope = encrypt_positions(snapshot, cert_response.json())
    ref = database.collection(PRIVATE_COLLECTION).document(PRIVATE_DOCUMENT)
    previous = ref.get()
    if previous.exists:
        old = previous.to_dict()
        if old.get("checked_at", "") > snapshot["checked_at"]:
            raise ValueError("Firebase 已有較新的持倉，不覆蓋")
    ref.set(envelope)
    result = ref.get()
    if not result.exists or result.to_dict().get("ciphertext") != envelope["ciphertext"]:
        raise RuntimeError("Firebase 寫入後核對失敗")


def encrypt_positions(snapshot, certificates):
    """Only a matching service-account private key can read the holdings."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    key, nonce = AESGCM.generate_key(bit_length=256), os.urandom(12)
    wrapped = {}
    for key_id, certificate in certificates.items():
        public = x509.load_pem_x509_certificate(certificate.encode()).public_key()
        wrapped[key_id] = base64.b64encode(public.encrypt(
            key, padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
        )).decode()
    if not wrapped:
        raise ValueError("Firebase 服務帳戶沒有可用的公開憑證")
    ciphertext = AESGCM(key).encrypt(
        nonce, json.dumps(snapshot, ensure_ascii=False).encode(), b"terawise:new_noble:v1"
    )
    return {"encrypted_schema": 1, "checked_at": snapshot["checked_at"],
            "wrapped_keys": wrapped, "nonce": base64.b64encode(nonce).decode(),
            "ciphertext": base64.b64encode(ciphertext).decode()}


def decrypt_positions(envelope, account):
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if envelope.get("encrypted_schema") != 1:
        raise ValueError("Firebase 持倉不是預期的加密格式")
    wrapped = envelope["wrapped_keys"].get(account["private_key_id"])
    if not wrapped:
        raise ValueError("同步與網站的 Firebase 憑證不相符，請使用同一服務帳戶並重新同步")
    private = serialization.load_pem_private_key(account["private_key"].encode(), password=None)
    key = private.decrypt(base64.b64decode(wrapped), padding.OAEP(
        mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None
    ))
    content = AESGCM(key).decrypt(
        base64.b64decode(envelope["nonce"]), base64.b64decode(envelope["ciphertext"]),
        b"terawise:new_noble:v1",
    )
    return json.loads(content)


def load_positions(database=None):
    if not os.environ.get("GITHUB_ACTIONS"):
        path = Path(os.environ.get("NEW_NOBLE_WORKBOOK", str(DEFAULT_WORKBOOK)))
        if path.exists():
            return read_positions(path)
    if database is None:
        raise ValueError("Firebase 未連線，無法讀取 New Noble 持倉")
    document = database.collection(PRIVATE_COLLECTION).document(PRIVATE_DOCUMENT).get()
    if not document.exists:
        raise ValueError("New Noble 持倉尚未同步至 Firebase")
    raw = os.environ.get("FIREBASE_SERVICE_ACCOUNT")
    if not raw:
        raise ValueError("網站缺少 Firebase 解密憑證")
    return decrypt_positions(document.to_dict(), json.loads(raw))


def _price(frame, date, sid):
    value = float(frame.at[pd.Timestamp(date), sid])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{pd.Timestamp(date):%Y-%m-%d} {sid} 價格缺漏")
    return value


def build_plan(snapshot, *, signal_date, buy_date, raw_close, raw_open, adj_close,
               candidates, daily_quota, moving_average, signal_close,
               benchmark_close, benchmark_average, now=None):
    """Symbols only: website entry rules, actual holdings, client 10%/40-day exits."""
    now = now or datetime.now(TAIPEI)
    signal_date, buy_date = pd.Timestamp(signal_date), pd.Timestamp(buy_date)
    if snapshot.get("schema_version") != 1 or snapshot.get("client") != "New Noble":
        raise ValueError("New Noble 持倉格式不符")
    checked = datetime.fromisoformat(snapshot["checked_at"])
    if checked.tzinfo is None or not timedelta(0) <= now - checked <= timedelta(hours=36):
        raise ValueError("持倉同步超過 36 小時或時間異常，請重新同步")
    if checked.astimezone(TAIPEI).date() < signal_date.date():
        raise ValueError("持倉尚未確認到訊號日，請重新同步")
    if pd.Timestamp(snapshot["latest_trade_date"]) > signal_date:
        raise ValueError("交易紀錄晚於行情資料日期，請等待最新收盤資料")
    if signal_date >= buy_date:
        raise ValueError("隔日交易日必須晚於訊號日")
    positions = snapshot["positions"]
    calendar = raw_close.index[raw_close.index <= signal_date]
    if signal_date not in calendar:
        raise ValueError("缺少訊號日行情")
    sells, notes = [], []
    available_cash = float(snapshot["cash"])
    if not math.isfinite(available_cash):
        raise ValueError("客戶現金餘額無效")
    for sid, p in positions.items():
        if int(p["shares"]) <= 0 or pd.Timestamp(p["entry_date"]) > signal_date:
            raise ValueError(f"{sid} 持倉資料無效")
        if not math.isclose(sum(lot["shares"] for lot in p["lots"]), p["shares"]):
            raise ValueError(f"{sid} 停損基準股數不符")
        current = _price(raw_close, signal_date, sid)
        current_adj = _price(adj_close, signal_date, sid)
        basis = sum(
            _price(raw_open, lot["date"], sid)
            * _price(adj_close, lot["date"], sid)
            / _price(raw_close, lot["date"], sid) * lot["shares"]
            for lot in p["lots"]
        ) / p["shares"]
        stop_return = current_adj / basis - 1
        entry = pd.Timestamp(p["entry_date"])
        if entry not in calendar:
            raise ValueError(f"{sid} 買進日不在行情索引")
        held_days = calendar.get_loc(signal_date) - calendar.get_loc(entry)
        reason = None
        if current_adj <= basis * 0.90:
            reason = f"停損（{stop_return:+.2%}）"
        elif held_days >= 40:
            reason = f"持有滿 {held_days} 交易日"
        if reason:
            sells.append({"stock_id": sid, "reason": reason})
            gross = current * p["shares"]
            available_cash += gross - max(20, math.floor(gross * 0.001425 + 0.5)) - math.floor(gross * 0.003 + 0.5)
    quota = max(0, int(daily_quota)) if 10 <= buy_date.day <= 15 else 0
    slots = min(max(0, 15 - len(positions) + len(sells)), quota)
    buys = []
    buy_blocked_reason = ""
    if slots:
        if not all(math.isfinite(float(v)) and float(v) > 0 for v in (benchmark_close, benchmark_average)):
            buy_blocked_reason = "大盤資料未更新"
            notes.append("大盤濾網資料缺漏，暫停新增買入；賣出照常檢查")
        elif benchmark_close < benchmark_average:
            notes.append("大盤低於 200 日均線，暫不新增買入")
        elif available_cash <= 0:
            notes.append("可用現金不足，暫不新增買入")
        elif candidates is None:
            raise ValueError("缺少訊號日選股排名")
        else:
            for sid in dict.fromkeys(_sid(s) for s in candidates):
                # Keep current holdings excluded even when scheduled for sale.
                if sid in positions:
                    continue
                last_buy = snapshot["last_buy_dates"].get(sid)
                if last_buy and len(calendar[(calendar > pd.Timestamp(last_buy))]) <= 5:
                    continue
                price, average = float(signal_close.get(sid, float("nan"))), float(moving_average.get(sid, float("nan")))
                if not math.isfinite(price) or not math.isfinite(average) or price <= average:
                    continue
                _price(raw_close, signal_date, sid)
                buys.append({"stock_id": sid, "reason": "分批進場、補足空位"})
                if len(buys) == slots:
                    break
    elif not 10 <= buy_date.day <= 15:
        notes.append("非每月 10–15 日買進窗口")
    elif len(positions) - len(sells) >= 15:
        notes.append("客戶持倉已達 15 檔")
    return {"signal_date": signal_date, "buy_date": buy_date, "sells": sells,
            "buys": buys, "notes": notes, "holdings_count": len(positions),
            "buy_blocked_reason": buy_blocked_reason,
            "checked_at": checked.astimezone(TAIPEI).strftime("%m/%d %H:%M")}


def format_plan(plan, name_mapper):
    esc = html.escape
    lines = []
    for key, label in (("sells", "🔴 賣出"), ("buys", "🟢 買入")):
        orders = plan.get(key, [])
        if not orders:
            continue
        lines.append(f"\n<b>{label}｜{plan['buy_date']:%m/%d}</b>")
        for order in orders:
            sid = order["stock_id"]
            name = esc(str(name_mapper.get(sid, "")))
            reason = esc(str(order["reason"]))
            lines.append(f"• {esc(sid)} {name}｜{reason}")
    if plan.get("buy_blocked_reason"):
        lines.append("\n⚠️ 買入暫緩：" + esc(plan["buy_blocked_reason"]) + "。")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--credentials", type=Path)
    parser.add_argument("--sync", action="store_true")
    args = parser.parse_args()
    snapshot = read_positions(args.workbook)
    print(f"New Noble｜最新交易 {snapshot['latest_trade_date']}｜持倉 {len(snapshot['positions'])} 檔")
    for sid, p in snapshot["positions"].items():
        print(f"  {sid}: {p['shares']:,} 股，首次進場 {p['entry_date']}")
    if args.sync:
        sync_positions(snapshot, get_database(args.credentials))
        print("Firebase 私有持倉同步成功；未發送 Telegram。")
    else:
        print("本機唯讀預覽完成；未寫入 Firebase、交易表或通知紀錄。")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        print(f"未完成：{error}")
        raise SystemExit(1)
