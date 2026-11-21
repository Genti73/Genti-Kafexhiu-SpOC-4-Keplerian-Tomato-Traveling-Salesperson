import sys
import time
import numpy as np
from multiprocessing import Pool, cpu_count
from utils_users import TomatoProblem

INSTANCE = "problems/easy.kttsp"
WHICH = sys.argv[1] if len(sys.argv) > 1 else "easy"
T_END_NEW = float(sys.argv[2]) if len(sys.argv) > 2 else 150.0
SRC = "edge_table_easy_fine.npz" if WHICH == "easy" else "edge_table_exc_fine.npz"
LIMIT = 100.0 if WHICH == "easy" else 600.0

_d = np.load(SRC)
T_OLD, TOF_OLD = _d["T"], _d["TOF"]
DT = float(T_OLD[1] - T_OLD[0])
T_NEW = np.arange(0.0, T_END_NEW, DT)
if WHICH == "easy":
    TOF_MAX_NEW = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
    TOF_NEW = np.arange(float(TOF_OLD[0]), TOF_MAX_NEW + 1e-9, DT)
else:
    TOF_NEW = TOF_OLD

util = None


def init_worker():
    global util
    util = TomatoProblem(INSTANCE)


def work(args):
    i, j, row = args
    n_old, m_old = len(T_OLD), len(TOF_OLD)
    out = np.full(len(T_NEW), -1, dtype=np.int16)
    out[:n_old] = row
    for k in range(len(T_NEW)):
        if k < n_old and row[k] >= 0:
            continue
        m_start = m_old if k < n_old else 0
        t = float(T_NEW[k])
        for m in range(m_start, len(TOF_NEW)):
            if util.compute_transfer(i, j, t, float(TOF_NEW[m])) <= LIMIT:
                out[k] = m
                break
    return i, j, out


def main():
    if not (np.allclose(T_NEW[:len(T_OLD)], T_OLD) and np.allclose(TOF_NEW[:len(TOF_OLD)], TOF_OLD)):
        print("GABIM: rrjetat e reja nuk e vazhdojnë saktë të vjetrën.")
        return
    pairs = [(int(i), int(j)) for i, j in _d["pairs"]]
    rows = _d["tof_idx"]
    n_proc = max(1, cpu_count() - 1)
    print(f"Tabela '{WHICH}': {len(pairs)} çifte | nisje: {len(T_OLD)} -> {len(T_NEW)} | "
          f"fluturime: {len(TOF_OLD)} -> {len(TOF_NEW)} | procese: {n_proc}")
    print("Po llogaritet... Mund ta ndalosh me Ctrl+C (asgjë nuk ruhet pa mbaruar).")

    index = {p: k for k, p in enumerate(pairs)}
    new_idx = np.full((len(pairs), len(T_NEW)), -1, dtype=np.int16)
    start = time.time()
    done = 0
    tasks = [(i, j, rows[index[(i, j)]]) for (i, j) in pairs]
    with Pool(processes=n_proc, initializer=init_worker) as pool:
        for i, j, out in pool.imap_unordered(work, tasks, chunksize=1):
            new_idx[index[(i, j)]] = out
            done += 1
            if done % 10 == 0 or done == len(pairs):
                el = time.time() - start
                print(f"  {done}/{len(pairs)} gati ({el:.0f} s, mbeten rreth {el / done * (len(pairs) - done):.0f} s)")

    outname = f"edge_table_{WHICH}_fine2.npz"
    np.savez_compressed(outname, pairs=np.array(pairs), tof_idx=new_idx, T=T_NEW, TOF=TOF_NEW)
    old_av = (rows >= 0).mean(axis=1)
    new_av = (new_idx[:, :len(T_OLD)] >= 0).mean(axis=1)
    print(f"\nU ruajt: {outname}")
    print(f"Disponueshmëria në nisjet e vjetra: mediane {np.median(old_av):.3f} -> {np.median(new_av):.3f} | "
          f"minimale {old_av.min():.3f} -> {new_av.min():.3f}")
    gained = int((new_av > old_av + 1e-9).sum())
    print(f"Çifte që fituan disponueshmëri: {gained} nga {len(pairs)}")
    print(f"Koha totale: {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()