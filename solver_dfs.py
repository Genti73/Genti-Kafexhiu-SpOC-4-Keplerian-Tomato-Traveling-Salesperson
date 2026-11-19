import json
import random
import sys
import time
import numpy as np
from utils_users import TomatoProblem
import validator
from solver import Table, decode, load_instance, PathGen, bits, INSTANCE

TIME_LIMIT = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0
BUDGET = int(float(sys.argv[2])) if len(sys.argv) > 2 else 30000


class Abort(Exception):
    pass


def feasible(gen, U, c, left):
    """Prerje strukturore: a mund të mbulohen ende domatet e mbetura?"""
    if gen.comps(U | (1 << c)) - 1 > left:
        return False
    avail = U | (1 << c)
    dead = 0
    for u in bits(U):
        if (gen.pe[u] | (gen.px[u] if left > 0 else 0)) & avail == 0:
            return False
        if gen.sa[u] & U == 0:
            dead += 1
            if dead > 1:
                return False
    return True


def main():
    N, t0, t_max, E = load_instance()
    easy, exc = Table("edge_table_easy.npz"), Table("edge_table_exc.npz")
    gen = PathGen(N, list(easy.rows), list(exc.rows), E)

    min_in = [float("inf")] * N
    for tab in (easy, exc):
        for (i, j), (_, _, idx) in tab.rows.items():
            min_in[j] = min(min_in[j], float(tab.TOF[idx[idx >= 0]].min()))
    print(f"Fluturimi minimal që hyn në një domate: nga {min(min_in):.2f} në {max(min_in):.2f} ditë")

    fail = {}
    best = {"f": t_max + 1e-9, "path": None, "kinds": None}
    deep = {"n": 0, "t": 0.0, "path": [], "kinds": []}
    st = {"exp": 0, "noise": 0.0, "deadline": 0.0}
    t_begin = time.time()

    def dfs(c, t, vis, used, lb, path, kinds):
        st["exp"] += 1
        if st["exp"] > BUDGET or time.time() > st["deadline"]:
            raise Abort
        d = len(path)
        if d > deep["n"] or (d == deep["n"] and t < deep["t"]):
            deep["n"], deep["t"], deep["path"], deep["kinds"] = d, t, list(path), list(kinds)
        U = gen.full & ~vis
        if U == 0:
            if t < best["f"]:
                best["f"], best["path"], best["kinds"] = t, list(path), list(kinds)
                print(f"  [{time.time() - t_begin:7.1f} s] f = {t:.3f} ditë")
            return
        if t + lb >= best["f"]:
            return
        key = (c, vis, used)
        ft = fail.get(key)
        if ft is not None and t >= ft - 1e-9:
            return
        left = E - used
        if not feasible(gen, U, c, left):
            fail[key] = 0.0
            return
        cands = []
        for j in bits(gen.se[c] & U):
            r = easy.hop(c, j, t)
            if r is not None:
                t2 = max(r[0], t) + r[1]
                if t2 <= t_max:
                    cands.append((t2 + random.random() * st["noise"], t2, j, 0))
        if left > 0:
            for j in bits(gen.sx[c] & U):
                r = exc.hop(c, j, t)
                if r is not None:
                    t2 = max(r[0], t) + r[1]
                    if t2 <= t_max:
                        cands.append((t2 + random.random() * st["noise"], t2, j, 1))
        cands.sort()
        for _, t2, j, k in cands:
            path.append(j)
            kinds.append(k)
            dfs(j, t2, vis | (1 << j), used + k, lb - min_in[j], path, kinds)
            path.pop()
            kinds.pop()
        if ft is None or t < ft:
            fail[key] = t

    lb_all = sum(min_in)
    deadline = t_begin + TIME_LIMIT
    restarts = total_exp = trivial = 0
    exhausted = False
    last_print = t_begin
    print(f"N = {N} | t_max = {t_max} | E = {E} | kufiri: {TIME_LIMIT:.0f} s | buxheti për rinisje: {BUDGET}")
    while time.time() < deadline:
        s = random.choice(gen.starts)
        st["exp"], st["deadline"] = 0, deadline
        st["noise"] = random.choice([0.0, 0.5, 1.0, 2.0])
        try:
            dfs(s, t0, 1 << s, 0, lb_all - min_in[s], [s], [])
        except Abort:
            pass
        total_exp += st["exp"]
        restarts += 1
        trivial = trivial + 1 if st["exp"] <= 1 else 0
        if len(fail) > 3000000:
            fail.clear()
        if trivial >= 300:
            exhausted = True
            break
        if time.time() - last_print > 10:
            last_print = time.time()
            print(f"  [{last_print - t_begin:6.0f} s] {restarts} rinisje | {total_exp} zgjerime | memo: {len(fail)} | "
                  f"thellësia më e madhe: {deep['n']}/{N} | f më e mirë: "
                  f"{'asnjë' if best['path'] is None else round(best['f'], 3)}")

    print()
    print(f"Rinisje: {restarts} | zgjerime: {total_exp} | memo: {len(fail)}")
    if exhausted and best["path"] is None:
        print("HAPËSIRA E KËRKIMIT U SHTERUA: nuk ekziston asnjë rrugë e plotë me këto tabela brenda t_max.")
    if best["path"] is None:
        allv = set(range(N))
        rem = sorted(allv - set(deep["path"]))
        print(f"Rruga më e thellë: {deep['n']}/{N} domate, koha = {deep['t']:.2f} ditë, "
              f"përjashtime = {sum(deep['kinds'])}")
        print("Domate të pavizituara:", rem)
        txt = [str(deep["path"][0])]
        for p, k in zip(deep["path"][1:], deep["kinds"]):
            txt.append(("=X=> " if k else "-> ") + str(p))
        print(" ".join(txt))
        return
    d = decode(best["path"], best["kinds"], easy, exc, t0, t_max)
    print(f"f më e mirë = {d[0]:.3f} ditë | përjashtime: {sum(best['kinds'])}")
    x = d[1] + d[2] + best["path"]
    with open("best_solution.json", "w", encoding="utf-8") as fh:
        json.dump({"x": [float(v) for v in x[:2 * (N - 1)]] + [int(v) for v in x[2 * (N - 1):]],
                   "f": d[0]}, fh)
    print("U ruajt në best_solution.json. Po e kontrolloj me validuesin...\n")
    validator.validate(x, TomatoProblem(INSTANCE), show_hops=False)


if __name__ == "__main__":
    main()