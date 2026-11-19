import json
import sys
import time
import numpy as np
from utils_users import TomatoProblem
import validator
from solver import Table, decode, load_instance, PathGen, bits, INSTANCE

K = int(float(sys.argv[1])) if len(sys.argv) > 1 else 50


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


def beam(N, t0, t_max, E, easy, exc, gen, low_mask, k_keep):
    states = {}
    for s in gen.starts:
        states[(s, 1 << s, 0)] = (t0, (s,), ())
    start = time.time()
    for level in range(1, N):
        cand = {}
        for (c, vis, used), (t, path, kinds) in states.items():
            U = gen.full & ~vis
            for j in bits(gen.se[c] & U):
                r = easy.hop(c, j, t)
                if r is None:
                    continue
                t2 = max(r[0], t) + r[1]
                if t2 > t_max:
                    continue
                key = (j, vis | (1 << j), used)
                if key not in cand or t2 < cand[key][0]:
                    cand[key] = (t2, path + (j,), kinds + (0,))
            if used < E:
                for j in bits(gen.sx[c] & U):
                    r = exc.hop(c, j, t)
                    if r is None:
                        continue
                    t2 = max(r[0], t) + r[1]
                    if t2 > t_max:
                        continue
                    key = (j, vis | (1 << j), used + 1)
                    if key not in cand or t2 < cand[key][0]:
                        cand[key] = (t2, path + (j,), kinds + (1,))
        groups = {}
        for key, val in cand.items():
            sig = ((key[1] & low_mask).bit_count(), key[2])
            groups.setdefault(sig, []).append((val[0], key, val))
        new = {}
        for sig, lst in groups.items():
            lst.sort(key=lambda z: z[0])
            kept = 0
            for t2, key, val in lst:
                c, vis, used = key
                if feasible(gen, gen.full & ~vis, c, E - used):
                    new[key] = val
                    kept += 1
                    if kept >= k_keep:
                        break
        states = new
        if not states:
            print(f"  Niveli {level + 1}: asnjë gjendje nuk mbijetoi. Beam-i vdiq këtu.")
            return None
        if (level + 1) % 6 == 0 or level == N - 1:
            ts = [v[0] for v in states.values()]
            print(f"  Niveli {level + 1:2d}/{N}: {len(states):5d} gjendje | koha më e mirë = {min(ts):7.2f} | "
                  f"më e keqja e mbajtur = {max(ts):7.2f} | {time.time() - start:.0f} s")
    return states


def main():
    N, t0, t_max, E = load_instance()
    easy, exc = Table("edge_table_easy.npz"), Table("edge_table_exc.npz")
    gen = PathGen(N, list(easy.rows), list(exc.rows), E)
    u = TomatoProblem(INSTANCE)
    low_mask = 0
    for i in range(N):
        if u.tomato_orbital_parameters[i, 0] < 8.0e6:
            low_mask |= 1 << i
    print(f"N = {N} | t_max = {t_max} | E = {E} | K për familje = {K}")
    states = beam(N, t0, t_max, E, easy, exc, gen, low_mask, K)
    if not states:
        print("Beam search nuk gjeti zgjidhje brenda kohës maksimale.")
        return
    t, path, kinds = min(states.values(), key=lambda v: v[0])
    d = decode(list(path), list(kinds), easy, exc, t0, t_max)
    print(f"\nf më e mirë = {d[0]:.3f} ditë | përjashtime: {sum(kinds)}")
    x = d[1] + d[2] + list(path)
    with open("best_solution.json", "w", encoding="utf-8") as fh:
        json.dump({"x": [float(v) for v in x[:2 * (N - 1)]] + [int(v) for v in x[2 * (N - 1):]],
                   "f": d[0]}, fh)
    print("U ruajt në best_solution.json. Po e kontrolloj me validuesin...\n")
    validator.validate(x, u, show_hops=False)


if __name__ == "__main__":
    main()