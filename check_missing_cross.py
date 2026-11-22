import sys
import time
import numpy as np
from multiprocessing import Pool, cpu_count
from utils_users import TomatoProblem

INSTANCE = "problems/easy.kttsp"
N_PAIRS = int(float(sys.argv[1])) if len(sys.argv) > 1 else 60
T_GRID = np.arange(0.0, 30.0, 0.02)
TOF_GRID = np.arange(0.2, 3.01, 0.02)
util = None


def init_worker():
    global util
    util = TomatoProblem(INSTANCE)


def work(pair):
    i, j = pair
    best = (np.inf, -1.0, -1.0)
    for t in T_GRID:
        t = float(t)
        for tof in TOF_GRID:
            dv = util.compute_transfer(i, j, t, float(tof))
            if dv < best[0]:
                best = (dv, t, float(tof))
    return i, j, best


def main():
    a = []
    with open(INSTANCE, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line and not line.startswith("c") and not line.startswith("p "):
                a.append(float(line.split()[0]))
    low = [v < 8.0e6 for v in a]
    N = len(a)
    coarse = np.load("min_dv_easy.npy")
    d = np.load("edge_table_exc.npz")
    have = {(int(i), int(j)) for (i, j), idx in zip(d["pairs"], d["tof_idx"]) if (idx >= 0).any()}
    cand = [(coarse[i, j], i, j) for i in range(N) for j in range(N)
            if i != j and low[i] != low[j] and (i, j) not in have]
    cand.sort()
    print(f"Çifte ulët<->lartë që nuk janë në tabelën e përjashtimeve: {len(cand)}")
    pairs = [(i, j) for _, i, j in cand[:N_PAIRS]]
    cmin = {(i, j): c for c, i, j in cand[:N_PAIRS]}
    print(f"Po skanohen {len(pairs)} (ato me dV më të ulët në skanimin e parë: {cand[0][0]:.0f} - {cand[len(pairs) - 1][0]:.0f})")
    print(f"Rrjeta: nisje 0-30 ditë (hapi 0.02) x fluturim 0.2-3.0 (hapi 0.02) | procese: {max(1, cpu_count() - 1)}")

    res = []
    start = time.time()
    with Pool(processes=max(1, cpu_count() - 1), initializer=init_worker) as pool:
        for i, j, best in pool.imap_unordered(work, pairs, chunksize=1):
            res.append((best[0], i, j, best[1], best[2]))
            if len(res) % 10 == 0 or len(res) == len(pairs):
                print(f"  {len(res)}/{len(pairs)} gati ({time.time() - start:.0f} s)")
    res.sort()
    ok = [r for r in res if r[0] <= 600.0]
    print(f"\nÇifte me dritare dV <= 600 të gjetura: {len(ok)} nga {len(pairs)}")
    print("10 më të mirat: dV e gjetur (skanimi i parë) | çifti | nisja | fluturimi")
    for r in res[:10]:
        print(f"  {r[0]:8.1f} ({cmin[(r[1], r[2])]:6.1f}) | {r[1]:2d}->{r[2]:2d} | t = {r[3]:.2f} | tof = {r[4]:.2f}")


if __name__ == "__main__":
    main()