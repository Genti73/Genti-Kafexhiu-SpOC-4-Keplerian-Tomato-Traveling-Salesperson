import json
import random
import sys
import time
from multiprocessing import Pool, cpu_count
from utils_users import TomatoProblem
import validator
from solver import Table, decode, load_instance, PathGen, bits, INSTANCE

HARVEST_S = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
M = int(float(sys.argv[2])) if len(sys.argv) > 2 else 100
F_COLLECT = float(sys.argv[3]) if len(sys.argv) > 3 else 150.0
STEP = 0.02
MARGIN = 0.02
BUDGET = 30000


class Abort(Exception):
    pass


def feasible(gen, U, c, left):
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


def harvest(N, t0, t_max, E, easy, exc, gen, seconds):
    """DFS që mbledh shumë renditje të plota me f (nga tabela) nën F_COLLECT."""
    cap = min(t_max, F_COLLECT)
    min_in = [float("inf")] * N
    for tab in (easy, exc):
        for (i, j), (_, _, idx) in tab.rows.items():
            min_in[j] = min(min_in[j], float(tab.TOF[idx[idx >= 0]].min()))
    found, fail = {}, {}
    st = {"exp": 0, "noise": 0.0, "deadline": 0.0}

    def dfs(c, t, vis, used, lb, path, kinds):
        st["exp"] += 1
        if st["exp"] > BUDGET or time.time() > st["deadline"]:
            raise Abort
        U = gen.full & ~vis
        if U == 0:
            key = tuple(path)
            if key not in found or t < found[key][0]:
                found[key] = (t, tuple(kinds))
            return True
        if t + lb >= cap:
            return False
        key = (c, vis, used)
        ft = fail.get(key)
        if ft is not None and t >= ft - 1e-9:
            return False
        left = E - used
        if not feasible(gen, U, c, left):
            fail[key] = 0.0
            return False
        cands = []
        for j in bits(gen.se[c] & U):
            r = easy.hop(c, j, t)
            if r is not None:
                t2 = max(r[0], t) + r[1]
                if t2 <= cap:
                    cands.append((t2 + random.random() * st["noise"], t2, j, 0))
        if left > 0:
            for j in bits(gen.sx[c] & U):
                r = exc.hop(c, j, t)
                if r is not None:
                    t2 = max(r[0], t) + r[1]
                    if t2 <= cap:
                        cands.append((t2 + random.random() * st["noise"], t2, j, 1))
        cands.sort()
        ok = False
        for _, t2, j, k in cands:
            path.append(j)
            kinds.append(k)
            if dfs(j, t2, vis | (1 << j), used + k, lb - min_in[j], path, kinds):
                ok = True
            path.pop()
            kinds.pop()
        if not ok and (ft is None or t < ft):
            fail[key] = t
        return ok

    lb_all = sum(min_in)
    deadline = time.time() + seconds
    last = time.time()
    while time.time() < deadline and len(found) < 20000:
        s = random.choice(gen.starts)
        st["exp"], st["deadline"] = 0, deadline
        st["noise"] = random.choice([0.0, 0.5, 1.0, 2.0])
        try:
            dfs(s, t0, 1 << s, 0, lb_all - min_in[s], [s], [])
        except Abort:
            pass
        if len(fail) > 3000000:
            fail.clear()
        if time.time() - last > 15:
            last = time.time()
            fs = sorted(v[0] for v in found.values())
            print(f"  [mbledhja] {len(found)} renditje | f më e mirë (tabelë) = {fs[0] if fs else '-'}")
    return found


_util = None


def init_worker():
    global _util
    _util = TomatoProblem(INSTANCE)


def polish_worker(args):
    path, kinds, t_old, tof_old, t0, f_table = args
    u = _util
    N = len(path)
    t_now = t0
    t_new, tof_new = [], []
    for k in range(N - 1):
        a, b = path[k], path[k + 1]
        lim = u.dv_exception if kinds[k] else u.dv_threshold
        best = (t_old[k] + tof_old[k], t_old[k], tof_old[k])
        i = 0
        while True:
            dep = t_now + i * STEP
            if dep >= best[0] - 1e-9:
                break
            tof = max(u.min_tof, STEP)
            while dep + tof < best[0] - 1e-9:
                if u.compute_transfer(a, b, dep, tof) <= lim - MARGIN:
                    best = (dep + tof, dep, tof)
                    break
                tof += STEP
            i += 1
        t_new.append(best[1])
        tof_new.append(best[2])
        t_now = best[1] + best[2]
    return t_now, t_new, tof_new, list(path), f_table


def main():
    N, t0, t_max, E = load_instance()
    easy, exc = Table("edge_table_easy.npz"), Table("edge_table_exc.npz")
    gen = PathGen(N, list(easy.rows), list(exc.rows), E)
    print(f"N = {N} | mbledhje: {HARVEST_S:.0f} s | do të polish-ohen: {M} | prag f (tabelë): {F_COLLECT}")
    found = harvest(N, t0, t_max, E, easy, exc, gen, HARVEST_S)
    items = sorted(found.items(), key=lambda kv: kv[1][0])
    print(f"Renditje të mbledhura: {len(items)}", end="")
    if not items:
        print("\nAsnjë renditje nën pragun F_COLLECT. Provo një prag më të lartë (argumenti i tretë).")
        return
    print(f" | f (tabelë): nga {items[0][1][0]:.2f} në {items[-1][1][0]:.2f}")

    tasks = []
    for path, (f_tab, kinds) in items[:M]:
        d = decode(list(path), list(kinds), easy, exc, t0, t_max)
        if d is not None:
            tasks.append((list(path), list(kinds), d[1], d[2], t0, f_tab))
    n_proc = max(1, cpu_count() - 1)
    print(f"Po polish-ohen {len(tasks)} renditje me {n_proc} procese...")

    results = []
    start = time.time()
    with Pool(processes=n_proc, initializer=init_worker) as pool:
        for r in pool.imap_unordered(polish_worker, tasks, chunksize=1):
            results.append(r)
            if len(results) % 10 == 0 or len(results) == len(tasks):
                b = min(x[0] for x in results)
                print(f"  {len(results)}/{len(tasks)} gati ({time.time() - start:.0f} s) | f më e mirë e polish-uar: {b:.3f}")

    results.sort(key=lambda r: r[0])
    print("\n8 më të mirat: f (tabelë) -> f (e polish-uar)")
    for r in results[:8]:
        print(f"  {r[4]:8.2f} -> {r[0]:8.3f}")
    f, t_list, tof_list, path = results[0][0], results[0][1], results[0][2], results[0][3]
    x = t_list + tof_list + path
    print()
    ok, fv, _, _ = validator.validate(x, TomatoProblem(INSTANCE), show_hops=False)
    if ok:
        out = f"best_harvest_f{fv:.3f}.json"
        json.dump({"x": [float(v) for v in x[:2 * (N - 1)]] + [int(v) for v in path], "f": fv}, open(out, "w"))
        print("U ruajt:", out)


if __name__ == "__main__":
    main()