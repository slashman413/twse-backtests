import json, os, sys
from datetime import datetime

BASE = "D:/TWSE-Data/Raw"
STATE_FILE = os.path.join(os.path.dirname(__file__), ".year_progress_state.json")

completed_years = {}
for y in range(2004, 2027):
    pf = os.path.join(BASE, f"_progress_{y}.json")
    if os.path.exists(pf):
        with open(pf) as f:
            d = json.load(f)
        dts = d.get("completed_dates", [])
        completed_years[y] = len(dts)

# load previous state
prev = {}
if os.path.exists(STATE_FILE):
    with open(STATE_FILE) as f:
        prev = json.load(f)

new_completed = []
for y, count in completed_years.items():
    prev_count = prev.get(str(y), 0)
    if prev_count > 0 and count > prev_count and count >= 240:
        new_completed.append((y, count))
    elif prev_count == 0 and count >= 240:
        new_completed.append((y, count))

# save state
with open(STATE_FILE, "w") as f:
    json.dump({str(k): v for k, v in completed_years.items()}, f)

if new_completed:
    msgs = [f"✅ {y} 年完成（{c}天）" for y, c in sorted(new_completed)]
    print("\n".join(msgs))
else:
    # also check if the latest year progress changed recently
    latest = max(completed_years.keys()) if completed_years else 0
    if latest:
        lc = completed_years[latest]
        lc_prev = prev.get(str(latest), 0)
        if lc != lc_prev and lc < 240:
            print(f"📊 {latest} 年: {lc}/{240}+ 天")
        else:
            print("no_change")
    else:
        print("no_change")
