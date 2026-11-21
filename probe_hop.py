import json
import sys
import numpy as np
from utils_users import TomatoProblem

SRC = sys.argv[1] if len(sys.argv) > 1 else "best_solution.json"
INSTANCE = "problems/easy.kttsp"
OFFSETS = (0.0, 0.013, 0.037)


def runs(ok, grid):
    out, i, n = [], 0, len(ok)
    while i < n:
        if ok[i]:
            j = i
            while j + 1 < n and ok[j + 1]:
                j += 1
            out.append((float(grid[i]), float(grid[j])))
            i = j
        i += 1
    return out


def hit(step, a, b):
    m = np.ceil(a / step - 1e-9) * step
    return m <= b + 1e-9


def main():
    u = TomatoProblem(INSTANCE)
    N = u.n_tomatoes
    x = json.load(open(SRC))["x"]
    t, tof = x[:N - 1], x[N - 1:2 * (N - 1)]
    pi = [int(v) for v in x[2 * (N - 1):]]
    grid = np.arange(0.05, 4.0, 0.001)
    widths, h05, h02, total = [], 0, 0, 0
    print("Për çdo kalim dhe zhvendosje të kohës së nisjes: dritaret e para të kohës së fluturimit ku dV <= kufirin")
    for k in range(2, N - 1, 6):
        a, b = pi[k], pi[k + 1]
        dv0 = u.compute_transfer(a, b, t[k], tof[k])
        lim = u.dv_threshold if dv0 <= u.dv_threshold else u.dv_exception
        for off in OFFSETS:
            dep = t[k] + off
            dv = np.array([u.compute_transfer(a, b, dep, float(g)) for g in grid])
            rs = runs(dv <= lim, grid)
            if not rs:
                print(f"  #{k + 1:2d} {a:2d}->{b:2d} (kufiri {lim:.0f}) zhvendosje {off:.3f}: asnjë dritare deri në 4 ditë")
                continue
            s, e = rs[0]
            w = e - s + 0.001
            widths.append(w)
            total += 1
            h05 += hit(0.05, s, e)
            h02 += hit(0.02, s, e)
            print(f"  #{k + 1:2d} {a:2d}->{b:2d} (kufiri {lim:.0f}) zhvendosje {off:.3f}: {len(rs)} dritare | "
                  f"e para: {s:.3f}-{e:.3f} (gjerësia {w:.3f}) | rrjeta 0.05: {'po' if hit(0.05, s, e) else 'JO'} | "
                  f"rrjeta 0.02: {'po' if hit(0.02, s, e) else 'JO'}")
    if total:
        print(f"\nGjerësia e dritares së parë: mediane = {np.median(widths):.3f} ditë | minimale = {min(widths):.3f}")
        print(f"E kap rrjeta 0.05: {h05}/{total} | e kap rrjeta 0.02: {h02}/{total}")


if __name__ == "__main__":
    main()