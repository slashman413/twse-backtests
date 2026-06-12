"""Fetch listed + OTC stock names from TWSE/TPEX OpenAPI and save to docs/data/stock_names.json."""
import json, os, ssl, urllib.request, sys

OUT_DIR = os.environ.get("BACKTEST_OUT_DIR",
          os.path.join(os.path.dirname(__file__), "..", "docs", "data"))
os.makedirs(OUT_DIR, exist_ok=True)

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))

names = {}

# TWSE listed companies
try:
    data = fetch_json("https://openapi.twse.com.tw/v1/opendata/t187ap03_L")
    for row in data:
        code = str(row.get("公司代號", "")).strip()
        name = str(row.get("公司名稱", "")).strip()
        if code and name:
            names[code.zfill(4)] = name
    print(f"TWSE listed: {len(names)} stocks")
except Exception as e:
    print(f"TWSE listed fetch failed: {e}")

# TPEX OTC companies
try:
    data = fetch_json("https://openapi.twse.com.tw/v1/opendata/t187ap03_O")
    before = len(names)
    for row in data:
        code = str(row.get("公司代號", "")).strip()
        name = str(row.get("公司名稱", "")).strip()
        if code and name:
            names[code.zfill(4)] = name
    print(f"TPEX OTC: {len(names) - before} stocks added")
except Exception as e:
    print(f"TPEX OTC fetch failed: {e}")

out_path = os.path.join(OUT_DIR, "stock_names.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(names, f, ensure_ascii=False, indent=2)

print(f"Total: {len(names)} stocks saved to {out_path}")
