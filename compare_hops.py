import json
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "best_solution.json"
sys.argv = sys.argv[:1]

import numpy as np
from utils_users import TomatoProblem
from solver import Table, load_instance, INSTANCE


def main():
    N, t0, t_max, E = load_instance()
    u = TomatoProblem(INSTANCE)
    x = json.load(open(SRC))["x"]
    t, tof = x[:N - 1], x[N - 1:2 * (N - 1)]
    pi = [int(v) for v in x[2 * (N - 1):]]
    a_sma = []
    with open(INSTANCE, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line and not line.startswith("c") and not line.startswith("p "):
                a_sma.append(float(line.split()[0]))
    low = [v < 8.0e6 for v in a_sma]
    easy, exc = Table("edge_table_easy_coarse.npz"), Table("edge_table_exc_coarse.npz")
    LB = np.load("easy_union.npz")["LB"]

    rows = []
    prev_arr = t0
    for k in range(N - 1):
        a, b = pi[k], pi[k + 1]
        dv = u.compute_transfer(a, b, t[k], tof[k])
        kind = 0 if dv <= u.dv_threshold else 1
        tab = exc if kind else easy
        r = tab.hop(a, b, prev_arr)
        pol_arr = t[k] + tof[k]
        if r is None:
            loss, avail = float("nan"), 0.0
        else:
            loss = max(r[0], prev_arr) + r[1] - pol_arr
            avail = float((tab.rows[(a, b)][2] >= 0).mean())
        grp = "përjashtim" if kind else ("e ulët" if (low[a] and low[b]) else "e lartë")
        rows.append((loss, k + 1, a, b, grp, t[k] - prev_arr, tof[k], float(LB[a, b]), avail))
        prev_arr = pol_arr

    fin = [r for r in rows if r[0] == r[0]]
    print(f"Humbja totale e tabelës së trashë (shuma e diferencave për kalim): {sum(r[0] for r in fin):.2f} ditë")
    for g in ("e lartë", "e ulët", "përjashtim"):
        sel = [r[0] for r in fin if r[4] == g]
        print(f"  {g:10s}: {len(sel):2d} kalime | humbja = {sum(sel):6.2f} ditë")
    print(f"  kalime ku tabela nuk ka asnjë mënyrë: {len(rows) - len(fin)}")
    print("\n12 kalimet me humbjen më të madhe:")
    print("   # nga  te  lloji        pritja   fluturimi   humbja    LB   disponueshmëria")
    for r in sorted(fin, key=lambda z: -z[0])[:12]:
        print(f"  {r[1]:2d} {r[2]:3d} {r[3]:3d}  {r[4]:10s} {r[5]:7.3f} {r[6]:10.3f} {r[0]:8.3f} {r[7]:6.1f} {r[8] * 100:8.0f}%")


if __name__ == "__main__":
    main()