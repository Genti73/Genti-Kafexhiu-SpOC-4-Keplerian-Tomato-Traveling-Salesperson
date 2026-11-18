import time
import numpy as np
from multiprocessing import Pool, cpu_count
from utils_users import TomatoProblem, MU_MOON

INSTANCE = "problems/easy.kttsp"

# Kërkim i gjerë: gjithë 200 ditët dhe fluturime deri në 5 ditë
T_GRID = np.arange(0.0, 200.0, 0.05)
TOF_GRID = np.arange(0.1, 5.01, 0.05)

util = None


def init_worker():
    global util
    util = TomatoProblem(INSTANCE)


def work(pair):
    """Kërko një transferim me dV <= 100; ndalo sapo ta gjesh."""
    i, j = pair
    best_dv, best_t, best_tof = np.inf, -1.0, -1.0
    for t in T_GRID:
        for tof in TOF_GRID:
            dv = util.compute_transfer(i, j, float(t), float(tof))
            if dv < best_dv:
                best_dv, best_t, best_tof = dv, float(t), float(tof)
            if dv <= 100.0:
                return i, j, best_dv, best_t, best_tof, True
    return i, j, best_dv, best_t, best_tof, False


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
    comps.sort(key=len, reverse=True)
    return comps


def report(easy, low, E, title):
    comps = components(easy | easy.T)
    print(title, "-> komponente:", len(comps))
    for k, c in enumerate(comps):
        tag = "ulët" if all(low[x] for x in c) else ("lartë" if not any(low[x] for x in c) else "përzier")
        print(f"  K{k}: madhësia {len(c):2d} ({tag}): {c}")
    print(f"  Kusht i nevojshëm: komponentet <= E + 1 = {E + 1} ->",
          "PLOTËSOHET" if len(comps) <= E + 1 else "NUK PLOTËSOHET")
    print()


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

    coarse = np.load("min_dv_easy.npy")
    fine = np.load("fine_easy.npz")["dv"]

    candidate = (LB <= dv_max) & ~eye
    coarse_easy = (coarse <= dv_max) & ~eye
    fine_easy = (fine <= dv_max) & ~eye
    known = coarse_easy | fine_easy

    print("Çifte kandidate (LB <= dV_max):          ", int(candidate.sum()))
    print("Të lehta sipas rrjetës së parë (e përafërt):", int(coarse_easy.sum()))
    print("Të lehta sipas rrjetës së dytë (e imët):    ", int(fine_easy.sum()))
    print("Të lehta sipas njërës ose tjetrës (union):  ", int(known.sum()))
    print("Të lehta në të parën por JO në të dytën:    ", int((coarse_easy & ~fine_easy).sum()))
    print()
    report(known, low, E, "Graf me çiftet e njohura të lehta (union)")

    uncertain = candidate & ~known
    pairs = [(i, j) for i in range(N) for j in range(N) if uncertain[i, j]]
    n_proc = max(1, cpu_count() - 1)
    print(f"Çifte të pasigurta (kandidate por ende pa gjetur transferim të lehtë): {len(pairs)}")
    print(f"Kërkim i gjerë: {len(T_GRID)} kohë nisjeje x {len(TOF_GRID)} kohë fluturimi, procese: {n_proc}")
    print("Po llogaritet... (ndalon për çdo çift sapo gjen transferim të lehtë)")

    final_dv = np.minimum(coarse, fine)
    final_t = np.full((N, N), -1.0)
    final_tof = np.full((N, N), -1.0)
    found = np.zeros((N, N), dtype=bool)

    start = time.time()
    done = 0
    with Pool(processes=n_proc, initializer=init_worker) as pool:
        for i, j, dv, t, tof, ok in pool.imap_unordered(work, pairs, chunksize=1):
            final_dv[i, j] = min(final_dv[i, j], dv)
            if ok:
                found[i, j] = True
                final_t[i, j], final_tof[i, j] = t, tof
            done += 1
            if done % 5 == 0 or done == len(pairs):
                print(f"  {done}/{len(pairs)} gati ({time.time() - start:.0f} s), gjetur të lehta: {int(found.sum())}")

    final_easy = known | found
    np.savez("easy_union.npz", easy=final_easy, dv=final_dv, t=final_t, tof=final_tof, LB=LB)

    print()
    print("Çifte të lehta gjithsej tani:", int(final_easy.sum()), "nga", int(candidate.sum()), "kandidate")
    print("Kandidate që mbeten të pa gjetura edhe pas kërkimit të gjerë:",
          [(i, j) for (i, j) in pairs if not found[i, j]])
    print()
    report(final_easy, low, E, "Graf përfundimtar i kalimeve të lehta")

    out_deg = final_easy.sum(axis=1)
    in_deg = final_easy.sum(axis=0)
    print("Pasues të lehtë (out):  ", [int(x) for x in out_deg])
    print("Paraardhës të lehtë (in):", [int(x) for x in in_deg])
    print("Pa pasues të lehtë:   ", [i for i in range(N) if out_deg[i] == 0])
    print("Pa paraardhës të lehtë:", [i for i in range(N) if in_deg[i] == 0])


if __name__ == "__main__":
    main()