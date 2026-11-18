import time
import numpy as np
from multiprocessing import Pool, cpu_count
from utils_users import TomatoProblem, MU_MOON

INSTANCE = "problems/easy.kttsp"

# Dritare kohore: 5 dritare nga 4 ditë, të shpërndara në 200 ditë
T_GRID = np.concatenate([np.arange(s, s + 4.0, 0.025) for s in range(0, 200, 40)])
TOF_GRID = np.arange(0.1, 3.0, 0.05)

util = None


def init_worker():
    global util
    util = TomatoProblem(INSTANCE)


def work(pair):
    i, j = pair
    best_dv, best_t, best_tof = np.inf, -1.0, -1.0
    n_easy = 0
    for t in T_GRID:
        for tof in TOF_GRID:
            dv = util.compute_transfer(i, j, float(t), float(tof))
            if dv <= 100.0:
                n_easy += 1
            if dv < best_dv:
                best_dv, best_t, best_tof = dv, float(t), float(tof)
    return i, j, best_dv, best_t, best_tof, n_easy


def load_instance():
    rows, params = [], None
    with open(INSTANCE, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("c"):
                continue
            if line.startswith("p "):
                params = line.split()[2:]
                continue
            rows.append([float(x) for x in line.split()])
    return np.array(rows), params


def components(adj):
    n = adj.shape[0]
    seen = [-1] * n
    comps = []
    for s in range(n):
        if seen[s] != -1:
            continue
        seen[s] = len(comps)
        comp = [s]
        stack = [s]
        while stack:
            u = stack.pop()
            for v in np.nonzero(adj[u])[0]:
                v = int(v)
                if seen[v] == -1:
                    seen[v] = len(comps)
                    comp.append(v)
                    stack.append(v)
        comps.append(sorted(comp))
    return comps


def main():
    orb, params = load_instance()
    N = len(orb)
    dv_max = float(params[3])
    E = int(float(params[5]))

    a, e, inc, raan = orb[:, 0], orb[:, 1], orb[:, 2], orb[:, 3]
    low = a < 8.0e6
    normals = np.stack([np.sin(inc) * np.sin(raan),
                        -np.sin(inc) * np.cos(raan),
                        np.cos(inc)], axis=1)
    H = normals * np.sqrt(MU_MOON * a * (1.0 - e ** 2))[:, None]
    apo = a * (1.0 + e)
    LB = np.linalg.norm(H[:, None, :] - H[None, :, :], axis=2) / np.maximum(apo[:, None], apo[None, :])
    np.fill_diagonal(LB, 0.0)

    eye = np.eye(N, dtype=bool)
    candidates = [(i, j) for i in range(N) for j in range(N)
                  if i != j and LB[i, j] <= dv_max]
    n_proc = max(1, cpu_count() - 1)
    print(f"Çifte kandidate (LB <= {dv_max}): {len(candidates)}")
    print(f"Pika për çift: {len(T_GRID) * len(TOF_GRID)}  |  procese: {n_proc}")
    print("Po llogaritet... (mund të marrë disa minuta)")

    mindv = np.full((N, N), np.inf)
    tbest = np.full((N, N), -1.0)
    tofbest = np.full((N, N), -1.0)
    neasy = np.zeros((N, N), dtype=int)

    start = time.time()
    done = 0
    with Pool(processes=n_proc, initializer=init_worker) as pool:
        for i, j, dv, t, tof, n in pool.imap_unordered(work, candidates, chunksize=1):
            mindv[i, j], tbest[i, j], tofbest[i, j], neasy[i, j] = dv, t, tof, n
            done += 1
            if done % 10 == 0 or done == len(candidates):
                print(f"  {done}/{len(candidates)} gati ({time.time() - start:.0f} s)")

    np.savez("fine_easy.npz", dv=mindv, t=tbest, tof=tofbest, n_easy=neasy, LB=LB)

    easy = (mindv <= dv_max) & ~eye
    print()
    print("Çifte kandidate:", len(candidates))
    print("Çifte që dolën vërtet të lehta (dV <= dV_max):", int(easy.sum()))
    print("Çifte kandidate që NUK dolën të lehta:        ", len(candidates) - int(easy.sum()))
    ratios = [mindv[i, j] / LB[i, j] for i, j in candidates if LB[i, j] > 1.0]
    print("Raporti dV_min_gjetur / LB: mediane =", round(float(np.median(ratios)), 3),
          " maksimale =", round(float(np.max(ratios)), 3))

    comps = components(easy | easy.T)
    comps.sort(key=len, reverse=True)
    print()
    print("Komponente në grafin e lehtë të rafinuar:", len(comps))
    for k, c in enumerate(comps):
        tag = "ulët" if all(low[x] for x in c) else ("lartë" if not any(low[x] for x in c) else "përzier")
        print(f"  K{k}: madhësia {len(c):2d} ({tag}): {c}")
    print(f"Kusht i nevojshëm: komponentet <= E + 1 = {E + 1} ->",
          "PLOTËSOHET" if len(comps) <= E + 1 else "NUK PLOTËSOHET")

    out_deg = easy.sum(axis=1)
    in_deg = easy.sum(axis=0)
    print()
    print("Pasues të lehtë (out):  ", [int(x) for x in out_deg])
    print("Paraardhës të lehtë (in):", [int(x) for x in in_deg])
    print("Pa pasues të lehtë:   ", [i for i in range(N) if out_deg[i] == 0])
    print("Pa paraardhës të lehtë:", [i for i in range(N) if in_deg[i] == 0])


if __name__ == "__main__":
    main()