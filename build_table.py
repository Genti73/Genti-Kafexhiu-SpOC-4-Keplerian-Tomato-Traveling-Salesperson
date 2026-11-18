import time
import numpy as np
from multiprocessing import Pool, cpu_count
from utils_users import TomatoProblem

INSTANCE = "problems/easy.kttsp"
DV_LIMIT = 100.0
T_END = 200.0
DT = 0.05
T_GRID = np.arange(0.0, T_END, DT)
TOF_GRID = np.arange(0.1, 5.01, 0.05)

util = None


def init_worker():
    global util
    util = TomatoProblem(INSTANCE)


def work(pair):
    """Për çdo kohë nisjeje, gjej indeksin e fluturimit më të shkurtër me dV <= limit (-1 = asnjë)."""
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
    data = np.load("easy_union.npz")
    easy = data["easy"]
    pairs = [(int(i), int(j)) for i, j in zip(*np.nonzero(easy))]
    index = {p: n for n, p in enumerate(pairs)}
    n_proc = max(1, cpu_count() - 1)

    print(f"Çifte të lehta: {len(pairs)} | kohë nisjeje: {len(T_GRID)} | fluturime: {len(TOF_GRID)} | procese: {n_proc}")
    print("Po ndërtohet tabela... (mund të marrë 20-45 minuta)")

    tof_idx = np.full((len(pairs), len(T_GRID)), -1, dtype=np.int16)

    start = time.time()
    done = 0
    with Pool(processes=n_proc, initializer=init_worker) as pool:
        for i, j, out in pool.imap_unordered(work, pairs, chunksize=1):
            tof_idx[index[(i, j)]] = out
            done += 1
            if done % 5 == 0 or done == len(pairs):
                el = time.time() - start
                eta = el / done * (len(pairs) - done)
                print(f"  {done}/{len(pairs)} gati ({el:.0f} s, mbeten rreth {eta:.0f} s)")

    np.savez_compressed("edge_table_easy.npz",
                        pairs=np.array(pairs), tof_idx=tof_idx,
                        T=T_GRID, TOF=TOF_GRID)

    frac = (tof_idx >= 0).mean(axis=1)
    print()
    print("Pjesa e kohëve të nisjes ku kalimi është i mundur (për çdo çift):")
    print(f"  minimale = {frac.min():.3f} | mediane = {np.median(frac):.3f} | maksimale = {frac.max():.3f}")
    tofs = TOF_GRID[tof_idx[tof_idx >= 0]]
    print(f"Koha e fluturimit kur është e mundur: mediane = {np.median(tofs):.2f} ditë, "
          f"mesatare = {tofs.mean():.2f} ditë")
    empty = [pairs[n] for n in range(len(pairs)) if frac[n] == 0]
    print("Çifte pa asnjë kohë nisjeje të mundur në këtë rrjetë:", empty)
    low_avail = [(pairs[n], round(float(frac[n]), 3)) for n in np.argsort(frac)[:8]]
    print("8 çiftet me disponueshmërinë më të ulët:", low_avail)
    print(f"Koha totale: {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()