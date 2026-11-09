"""Worked example for the Select function in Algorithm 1 (Section 4.2).

Run from the paper folder: python3 scripts/select_example.py
"""

def select(pool, n_rsrv, max_bs, n_slots):
    cand = sorted((i for i, r in enumerate(pool) if r["state"] != "RUNNING"), key=lambda i: pool[i]["arrival"])
    batch = []
    for i in cand:
        if len(batch) == max_bs:
            break
        if pool[i]["state"] == "INITIATION":
            new = n_rsrv + pool[i]["max_tokens"]
            if new > n_slots:
                break
            n_rsrv = new
        batch.append(i)
    return batch, n_rsrv


pool = [
    {"name": "A", "arrival": 0, "state": "INCREMENT", "max_tokens": 40},
    {"name": "B", "arrival": 1, "state": "INITIATION", "max_tokens": 50},
    {"name": "C", "arrival": 2, "state": "INITIATION", "max_tokens": 30},
    {"name": "D", "arrival": 3, "state": "INITIATION", "max_tokens": 10},
]
n_slots, max_bs = 100, 4
n_rsrv = 40  # A already holds 40 slots
print(f"n_slots={n_slots}, A holds {n_rsrv}")
batch, new_rsrv = select(pool, n_rsrv, max_bs, n_slots)
print("batch =", [pool[i]["name"] for i in batch], "n_rsrv =", new_rsrv)
print("C needs 30 more: 40 + 50 + 30 =", 40 + 50 + 30, "> 100, so C waits")
print("D would fit alone (", 40 + 50 + 10, "<= 100 ) but FCFS blocks it behind C")
