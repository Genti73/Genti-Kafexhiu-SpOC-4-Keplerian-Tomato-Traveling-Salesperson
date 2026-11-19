import json
import random
import sys
import time
import numpy as np

E_MAX = 5
TIME_LIMIT = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
MAX_EXP = 200000          # zgjerime maksimale për çdo rinisje
INSTANCE = "problems/easy.kttsp"


class Abort(Exception):
    pass


def count_tomatoes():
    n = 0
    with open(INSTANCE, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line and not line.startswith("c") and not line.startswith("p "):
                n += 1
    return n


def bits(m):
    while m:
        b = m & -m
        yield b.bit_length() - 1
        m ^= b


def main():
    N = count_tomatoes()
    d = np.load("edge_table_easy.npz")
    easy = [(int(i), int(j)) for i, j in d["pairs"]]
    e = np.load("edge_table_exc.npz")
    exc = [(int(i), int(j)) for (i, j), row in zip(e["pairs"], e["tof_idx"]) if (row >= 0).any()]
    print(f"N = {N} | kalime të lehta: {len(easy)} | kalime përjashtimi të mundshme: {len(exc)}")

    succ_e = [0] * N
    succ_x = [0] * N
    pred_e = [0] * N
    pred_x = [0] * N
    und_e = [0] * N
    for i, j in easy:
        succ_e[i] |= 1 << j
        pred_e[j] |= 1 << i
        und_e[i] |= 1 << j
        und_e[j] |= 1 << i
    for i, j in exc:
        succ_x[i] |= 1 << j
        pred_x[j] |= 1 << i
    succ_all = [succ_e[i] | succ_x[i] for i in range(N)]
    full = (1 << N) - 1

    def n_components(mask):
        comps, rem = 0, mask
        while rem:
            low = rem & -rem
            comp, frontier = low, low
            while frontier:
                b = frontier & -frontier
                frontier ^= b
                new = und_e[b.bit_length() - 1] & rem & ~comp
                comp |= new
                frontier |= new
            rem &= ~comp
            comps += 1
        return comps

    state = {"exp": 0, "best": 0, "deadline": 0.0}

    def dfs(c, visited, used, path, kinds):
        state["exp"] += 1
        if state["exp"] > MAX_EXP or time.time() > state["deadline"]:
            raise Abort
        depth = len(path)
        if depth > state["best"]:
            state["best"] = depth
        if depth == N:
            return True
        U = full & ~visited
        left = E_MAX - used
        if n_components(U | (1 << c)) - 1 > left:
            return False
        avail = U | (1 << c)
        n_dead_end = 0
        for u in bits(U):
            pm = pred_e[u] | (pred_x[u] if left > 0 else 0)
            if pm & avail == 0:
                return False
            if succ_all[u] & U == 0:
                n_dead_end += 1
                if n_dead_end > 1:
                    return False
        cands = []
        for j in bits(succ_e[c] & U):
            cands.append((((succ_all[j] & U).bit_count()), random.random(), j, 0))
        if left > 0:
            for j in bits(succ_x[c] & U):
                cands.append((((succ_all[j] & U).bit_count()) + 100, random.random(), j, 1))
        cands.sort()
        for _, _, j, k in cands:
            path.append(j)
            kinds.append(k)
            if dfs(j, visited | (1 << j), used + k, path, kinds):
                return True
            path.pop()
            kinds.pop()
        return False

    starts = [u for u in range(N) if pred_e[u] | pred_x[u]]
    forced = [u for u in range(N) if (pred_e[u] | pred_x[u]) == 0]
    if len(forced) > 1:
        print("Më shumë se një domate pa asnjë paraardhës:", forced, "-> nuk ka renditje të mundshme")
        return
    if forced:
        starts = forced

    t_start = time.time()
    deadline = t_start + TIME_LIMIT
    restarts = 0
    overall_best = 0
    while time.time() < deadline:
        s = random.choice(starts)
        state["exp"], state["best"], state["deadline"] = 0, 0, deadline
        path, kinds = [s], []
        try:
            ok = dfs(s, 1 << s, 0, path, kinds)
        except Abort:
            ok = False
        restarts += 1
        overall_best = max(overall_best, state["best"])
        if ok:
            print(f"U GJET renditje pas {restarts} rinisjesh ({time.time() - t_start:.1f} s)")
            print(f"Përjashtime të përdorura: {sum(kinds)} nga {E_MAX}")
            txt = [str(path[0])]
            for p, k in zip(path[1:], kinds):
                txt.append(("=X=> " if k else "-> ") + str(p))
            print(" ".join(txt))
            with open("static_path.json", "w", encoding="utf-8") as fh:
                json.dump({"path": path, "kinds": kinds}, fh)
            print("U ruajt në static_path.json")
            return
        if restarts % 20 == 0:
            print(f"  {restarts} rinisje, thellësia më e mirë: {overall_best}/{N} ({time.time() - t_start:.0f} s)")
    print(f"NUK u gjet renditje në {TIME_LIMIT:.0f} s ({restarts} rinisje). Thellësia më e mirë: {overall_best}/{N}")


if __name__ == "__main__":
    main()