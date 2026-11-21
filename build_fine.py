import sys
import time
import numpy as np
from multiprocessing import Pool, cpu_count
from utils_users import TomatoProblem

INSTANCE = "problems/easy.kttsp"
WHICH = sys.argv[1] if len(sys.argv) > 1 else "easy"
T_END = float(sys.argv[2]) if len(sys.argv) > 2 else 125.0
DT = float(sys.argv[3]) if len(sys.argv) > 3 else 0.02
T_GRID = np.arange(0.0, T_END, DT)
if WHICH == "easy":
    LIMIT, TOF_GRID = 100.0, np.arange(0.1, 5.01, DT)
else:
    LIMIT, TOF_GRID = 600.0, np.arange(0.2, 3.01, DT)

util = None


def init_worker():
    global util
    util = TomatoProblem(INSTANCE)


def work(pair):
    """Për çdo kohë nisjeje, indeksi i fluturimit më të shkurtër me dV <= LIMIT (-1 = asnjë)."""
    i, j = pair
    out = np.full(len(T_GRID), -1, dtype=np.int16)
    for k, t in enumerate(T_GRID):
        t = float(t)
        for m, tof in enumerate(TOF_GRID):
            if util.compute_transfer(i, j, t, float(tof)) <= LIMIT:
                out[k] = m
                break
    return i, j, out


def main():
    if WHICH == "easy":
        d = np.load("easy_union.npz")
        LB = d["LB"]
        n = LB.shape[0]
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j and LB[i, j] <= 100.0]
        outname = "edge_table_easy_fine.npz"
    else:
        d = np.load("edge_table_exc.npz")
        pairs = [(int(i), int(j)) for (i, j), idx in zip(d["pairs"], d["tof_idx"]) if (idx >= 0).any()]
        outname = "edge_table_exc_fine.npz"
    n_proc = max(1, cpu_count() - 1)
    print(f"Tabela '{WHICH}': {len(pairs)} çifte | {len(T_GRID)} kohë nisjeje | {len(TOF_GRID)} fluturime | "
          f"hapi {DT} | procese: {n_proc}")
    print("Po ndërtohet... Mund ta ndalosh me Ctrl+C (asgjë nuk ruhet pa mbaruar).")

    index = {p: k for k, p in enumerate(pairs)}
    tof_idx = np.full((len(pairs), len(T_GRID)), -1, dtype=np.int16)
    start = time.time()
    done = 0
    with Pool(processes=n_proc, initializer=init_worker) as pool:
        for i, j, out in pool.imap_unordered(work, pairs, chunksize=1):
            tof_idx[index[(i, j)]] = out
            done += 1
            if done % 10 == 0 or done == len(pairs):
                el = time.time() - start
                print(f"  {done}/{len(pairs)} gati ({el:.0f} s, mbeten rreth {el / done * (len(pairs) - done):.0f} s)")

    np.savez_compressed(outname, pairs=np.array(pairs), tof_idx=tof_idx, T=T_GRID, TOF=TOF_GRID)
    frac = (tof_idx >= 0).mean(axis=1)
    print(f"\nU ruajt: {outname}")
    print(f"Çifte me të paktën një kohë të mundur: {int((frac > 0).sum())} nga {len(pairs)}")
    print(f"Disponueshmëria: minimale = {frac.min():.3f} | mediane = {np.median(frac):.3f} | maksimale = {frac.max():.3f}")
    print(f"Koha totale: {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()