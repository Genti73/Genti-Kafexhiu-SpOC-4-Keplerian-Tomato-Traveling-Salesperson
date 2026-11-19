import json
import math
import random
import sys
import time
import numpy as np
from utils_users import TomatoProblem
import validator

INSTANCE = "problems/easy.kttsp"
TIME_LIMIT = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
MAX_EXP = 20000


def load_instance():
    n, params = 0, None
    with open(INSTANCE, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("p "):
                params = line.split()[2:]
            elif line and not line.startswith("c"):
                n += 1
    return n, float(params[0]), float(params[2]), int(float(params[5]))


class Table:
    """Tabela e kalimeve: për çdo çift, mbërritja më e hershme kur nisesh në ose pas kohës t."""

    def __init__(self, path):
        d = np.load(path)
        self.T, self.TOF = d["T"], d["TOF"]
        self.DT = float(self.T[1] - self.T[0])
        self.rows = {}
        for (i, j), idx in zip(d["pairs"], d["tof_idx"]):
            if not (idx >= 0).any():
                continue
            arr = np.where(idx >= 0, self.T + self.TOF[np.maximum(idx, 0)], np.inf)
            sk = np.empty(len(arr), dtype=np.int32)
            best, best_k = np.inf, -1
            for k in range(len(arr) - 1, -1, -1):
                if arr[k] <= best:
                    best, best_k = arr[k], k
                sk[k] = best_k
            self.rows[(int(i), int(j))] = (arr[sk], sk, idx)

    def hop(self, i, j, t_now):
        r = self.rows.get((i, j))
        if r is None:
            return None
        best_arr, sk, idx = r
        k = max(0, int(math.ceil(t_now / self.DT - 1e-9)))
        while k < len(self.T) and self.T[k] < t_now - 1e-12:
            k += 1
        if k >= len(self.T) or not np.isfinite(best_arr[k]):
            return None
        kb = sk[k]
        return float(self.T[kb]), float(self.TOF[idx[kb]])      # (nisja, fluturimi)


def decode(path, kinds, easy, exc, t0, t_max):
    """Orari më i hershëm për një renditje të dhënë. Kthen (f, t, tof) ose None."""
    t_now, ts, tofs = t0, [], []
    for a, b, k in zip(path, path[1:], kinds):
        r = (exc if k else easy).hop(a, b, t_now)
        if r is None:
            return None
        dep, tof = r
        dep = max(dep, t_now)
        ts.append(dep)
        tofs.append(tof)
        t_now = dep + tof
        if t_now > t_max:
            return None
    return t_now, ts, tofs


class Abort(Exception):
    pass


def bits(m):
    while m:
        b = m & -m
        yield b.bit_length() - 1
        m ^= b


class PathGen:
    """Backtracking i rastësishëm me prerje (si në static_search.py)."""

    def __init__(self, N, easy_edges, exc_edges, E):
        self.N, self.E = N, E
        self.se, self.sx = [0] * N, [0] * N
        self.pe, self.px = [0] * N, [0] * N
        self.ue = [0] * N
        for i, j in easy_edges:
            self.se[i] |= 1 << j
            self.pe[j] |= 1 << i
            self.ue[i] |= 1 << j
            self.ue[j] |= 1 << i
        for i, j in exc_edges:
            self.sx[i] |= 1 << j
            self.px[j] |= 1 << i
        self.sa = [self.se[i] | self.sx[i] for i in range(N)]
        self.full = (1 << N) - 1
        self.starts = [u for u in range(N) if self.pe[u] | self.px[u]]

    def comps(self, mask):
        n, rem = 0, mask
        while rem:
            low = rem & -rem
            comp, fr = low, low
            while fr:
                b = fr & -fr
                fr ^= b
                new = self.ue[b.bit_length() - 1] & rem & ~comp
                comp |= new
                fr |= new
            rem &= ~comp
            n += 1
        return n

    def generate(self, noise, deadline):
        N, E = self.N, self.E
        st = {"exp": 0}

        def dfs(c, vis, used, path, kinds):
            st["exp"] += 1
            if st["exp"] > MAX_EXP or time.time() > deadline:
                raise Abort
            if len(path) == N:
                return True
            U = self.full & ~vis
            left = E - used
            if self.comps(U | (1 << c)) - 1 > left:
                return False
            avail = U | (1 << c)
            dead = 0
            for u in bits(U):
                if (self.pe[u] | (self.px[u] if left > 0 else 0)) & avail == 0:
                    return False
                if self.sa[u] & U == 0:
                    dead += 1
                    if dead > 1:
                        return False
            cands = []
            for j in bits(self.se[c] & U):
                cands.append(((self.sa[j] & U).bit_count() + random.random() * noise, j, 0))
            if left > 0:
                for j in bits(self.sx[c] & U):
                    cands.append(((self.sa[j] & U).bit_count() + 100 + random.random() * noise, j, 1))
            cands.sort()
            for _, j, k in cands:
                path.append(j)
                kinds.append(k)
                if dfs(j, vis | (1 << j), used + k, path, kinds):
                    return True
                path.pop()
                kinds.pop()
            return False

        s = random.choice(self.starts)
        path, kinds = [s], []
        try:
            return (path, kinds) if dfs(s, 1 << s, 0, path, kinds) else None
        except Abort:
            return None


def main():
    N, t0, t_max, E = load_instance()
    easy, exc = Table("edge_table_easy.npz"), Table("edge_table_exc.npz")
    gen = PathGen(N, list(easy.rows), list(exc.rows), E)
    print(f"N = {N} | t_max = {t_max} | E = {E} | kufiri kohor: {TIME_LIMIT:.0f} s")

    start = time.time()
    deadline = start + TIME_LIMIT
    n_paths = n_ok = 0
    fs = []
    best = None
    while time.time() < deadline:
        r = gen.generate(random.choice([0.0, 1.0, 2.0, 4.0]), deadline)
        if r is None:
            continue
        n_paths += 1
        d = decode(r[0], r[1], easy, exc, t0, t_max)
        if d is None:
            continue
        n_ok += 1
        fs.append(d[0])
        if best is None or d[0] < best[0]:
            best = (d[0], d[1], d[2], r[0], r[1])
            print(f"  [{time.time() - start:6.1f} s] renditja #{n_paths}: f = {best[0]:.3f} ditë")

    print()
    print(f"Renditje strukturore të gjetura: {n_paths} | me orar brenda {t_max:.0f} ditëve: {n_ok}")
    if not best:
        print("Asnjë renditje nuk doli e vlefshme me kohën.")
        return
    print(f"f më e mirë = {best[0]:.3f} | mediane = {np.median(fs):.3f} | më e keqja = {max(fs):.3f}")
    x = best[1] + best[2] + best[3]
    with open("best_solution.json", "w", encoding="utf-8") as fh:
        json.dump({"x": [float(v) for v in x[:2 * (N - 1)]] + [int(v) for v in x[2 * (N - 1):]],
                   "f": best[0]}, fh)
    print("U ruajt në best_solution.json. Po e kontrolloj me validuesin...")
    print()
    validator.validate(x, TomatoProblem(INSTANCE), show_hops=False)


if __name__ == "__main__":
    main()