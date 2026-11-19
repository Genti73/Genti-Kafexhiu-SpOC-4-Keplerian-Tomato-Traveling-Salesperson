import time
import numpy as np
from multiprocessing import Pool, cpu_count
from utils_users import TomatoProblem

INSTANCE = "problems/easy.kttsp"
DV_LIMIT = 600.0
T_GRID = np.arange(0.0, 200.0, 0.1)
TOF_GRID = np.arange(0.2, 3.01, 0.1)

util = None


def init_worker():
    global util
    util = TomatoProblem(INSTANCE)


def work(pair):
    """Për çdo kohë nisjeje, indeksi i fluturimit më të shkurtër me dV <= 600 (-1 = asnjë)."""
    i, j = pair
    out = np.full(len(T_GRID), -1, dtype=np.int16)
    for k, t in enumerate(T_GRID):
        t = float(t)
        for m, tof in enumerate(TOF_GRID):
            if util.compute_transfer(i, j, t, float(tof)) <= DV_LIMIT:
                out[k] = m
                break
    return i, j, out


def main():
    u = TomatoProblem(INSTANCE)
    low = u.tomato_orbital_parameters[:, 0] < 8.0e6
    N = len(low)
    pairs = [(i, j) for i in range(N) for j in range(N) if low[i] != low[j]]
    n_proc = max(1, cpu_count() - 1)
    print(f"Çifte ulët<->lartë: {len(pairs)} | kohë nisjeje: {len(T_GRID)} | fluturime: {len(TOF_GRID)} | procese: {n_proc}")
    print("Po ndërtohet tabela... (mund të marrë 15-40 minuta)")

    tof_idx = np.full((len(pairs), len(T_GRID)), -1, dtype=np.int16)
    index = {p: n for n, p in enumerate(pairs)}
    start = time.time()
    done = 0
    with Pool(processes=n_proc, initializer=init_worker) as pool:
        for i, j, out in pool.imap_unordered(work, pairs, chunksize=1):
            tof_idx[index[(i, j)]] = out
            done += 1
            if done % 40 == 0 or done == len(pairs):
                el = time.time() - start
                print(f"  {done}/{len(pairs)} gati ({el:.0f} s, mbeten rreth {el / done * (len(pairs) - done):.0f} s)")

    np.savez_compressed("edge_table_exc.npz", pairs=np.array(pairs), tof_idx=tof_idx,
                        T=T_GRID, TOF=TOF_GRID)

    frac = (tof_idx >= 0).mean(axis=1)
    ok = [n for n in range(len(pairs)) if frac[n] > 0]
    print()
    print(f"Çifte me të paktën një kohë të mundur: {len(ok)} nga {len(pairs)}")
    if ok:
        print(f"Disponueshmëria (pjesa e kohëve) mes tyre: minimale = {frac[ok].min():.3f} | "
              f"mediane = {np.median(frac[ok]):.3f} | maksimale = {frac[ok].max():.3f}")
    for name, sel in (("Nga e ulët -> e lartë", lambda p: low[p[0]]),
                      ("Nga e lartë -> e ulët", lambda p: low[p[1]])):
        print(name + ", çifte të mundura për çdo domate të ulët:")
        cnt = {}
        for n in ok:
            p = pairs[n]
            if sel(p):
                key = p[0] if low[p[0]] else p[1]
                cnt[key] = cnt.get(key, 0) + 1
        print("  ", {k: cnt.get(k, 0) for k in range(N) if low[k]})
    print(f"Koha totale: {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()