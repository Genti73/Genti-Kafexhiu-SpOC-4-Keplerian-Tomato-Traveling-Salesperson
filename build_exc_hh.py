import os
import shutil
import time
import numpy as np
from multiprocessing import Pool, cpu_count
import build_exc_table as B

try:
    from utils_users import MU_MOON
except Exception:
    MU_MOON = 4.9048695e12

LB_MAX = 200.0
INSTANCE = B.INSTANCE
CHECK = [(2, 12), (12, 2), (7, 14), (14, 7), (6, 24)]


def main():
    rows = []
    with open(INSTANCE, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line and not line.startswith("c") and not line.startswith("p "):
                rows.append([float(x) for x in line.split()])
    orb = np.array(rows)
    N = len(orb)
    a, e, inc, raan = orb[:, 0], orb[:, 1], orb[:, 2], orb[:, 3]
    low = a < 8.0e6
    normals = np.stack([np.sin(inc) * np.sin(raan), -np.sin(inc) * np.cos(raan), np.cos(inc)], axis=1)
    H = normals * np.sqrt(MU_MOON * a * (1.0 - e ** 2))[:, None]
    apo = a * (1.0 + e)
    LB = np.linalg.norm(H[:, None, :] - H[None, :, :], axis=2) / np.maximum(apo[:, None], apo[None, :])

    de = np.load("edge_table_easy.npz")
    easy = {(int(i), int(j)) for (i, j), idx in zip(de["pairs"], de["tof_idx"]) if (idx >= 0).any()}

    if not os.path.exists("edge_table_exc_cross.npz"):
        shutil.copy("edge_table_exc.npz", "edge_table_exc_cross.npz")
    base = np.load("edge_table_exc_cross.npz")
    if len(base["T"]) != len(B.T_GRID) or len(base["TOF"]) != len(B.TOF_GRID):
        print("GABIM: rrjeta e tabelës ekzistuese nuk përputhet me atë të skriptit.")
        return

    pairs = [(i, j) for i in range(N) for j in range(N)
             if i != j and not low[i] and not low[j] and LB[i, j] <= LB_MAX and (i, j) not in easy]
    n_proc = max(1, cpu_count() - 1)
    print(f"Çifte të larta jo-të-lehta me LB <= {LB_MAX}: {len(pairs)} | procese: {n_proc}")
    print("Po ndërtohet... (disa minuta)")

    index = {p: n for n, p in enumerate(pairs)}
    tof_idx = np.full((len(pairs), len(B.T_GRID)), -1, dtype=np.int16)
    start = time.time()
    done = 0
    with Pool(processes=n_proc, initializer=B.init_worker) as pool:
        for i, j, out in pool.imap_unordered(B.work, pairs, chunksize=1):
            tof_idx[index[(i, j)]] = out
            done += 1
            if done % 25 == 0 or done == len(pairs):
                el = time.time() - start
                print(f"  {done}/{len(pairs)} gati ({el:.0f} s, mbeten rreth {el / done * (len(pairs) - done):.0f} s)")

    np.savez_compressed("edge_table_exc.npz",
                        pairs=np.concatenate([base["pairs"], np.array(pairs).reshape(-1, 2)]),
                        tof_idx=np.concatenate([base["tof_idx"], tof_idx]),
                        T=base["T"], TOF=base["TOF"])

    frac = (tof_idx >= 0).mean(axis=1)
    ok = int((frac > 0).sum())
    print()
    print(f"Çifte të reja me të paktën një kohë të mundur: {ok} nga {len(pairs)}")
    if ok:
        print(f"Disponueshmëria mes tyre: mediane = {np.median(frac[frac > 0]):.2f}, maksimale = {frac.max():.2f}")
    print("Pesë çiftet që mungonin më parë (disponueshmëria si përjashtim):")
    for p in CHECK:
        print(f"  {p}: " + (f"{frac[index[p]] * 100:.1f}%" if p in index else "jashtë listës"))
    print("U përditësua edge_table_exc.npz (kopja e vjetër: edge_table_exc_cross.npz)")
    print(f"Koha totale: {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()