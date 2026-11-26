import bisect
import json
import os
import pickle
import sys
import time
import numpy as np
from multiprocessing import Pool, cpu_count
from utils_users import TomatoProblem
import validator

INSTANCE = sys.argv[1] if len(sys.argv) > 1 else "problems/easy.kttsp"
K_FAM = int(float(sys.argv[2])) if len(sys.argv) > 2 else 3
K_EASY = int(float(sys.argv[3])) if len(sys.argv) > 3 else 5
M_EXC = int(float(sys.argv[4])) if len(sys.argv) > 4 else 2
ALPHA = float(sys.argv[5]) if len(sys.argv) > 5 else 3.0
RESUME = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6] != "-" else None
DELTA = float(sys.argv[7]) if len(sys.argv) > 7 else 0.04
SWEEP = int(float(sys.argv[8])) if len(sys.argv) > 8 else 0
BACK = float(sys.argv[9]) if len(sys.argv) > 9 else 0.06
START_TOL = float(sys.argv[10]) if len(sys.argv) > 10 else 0.08
MU = 4.9048695e12
MARGIN = 0.02

util = None
LOW = None


def init_worker(low):
    global util, LOW
    util = TomatoProblem(INSTANCE)
    LOW = low


def eval_hop(task):
    """Mbërritja më e hershme për a->b duke u nisur në ose pas t_now, me dV <= lim - MARGIN."""
    a, b, t_now, lim, kind = task
    if kind == 1:
        ds, ts, tof_max, wait_max = 0.05, 0.02, 3.0, 6.0
    elif LOW[a] and LOW[b]:
        ds, ts, tof_max, wait_max = 0.01, 0.01, 2.0, 2.0
    else:
        ds, ts, tof_max, wait_max = 0.05, 0.05, 8.0, 3.0
    tof_min = max(util.min_tof, ts)
    best = None
    i = 0
    while True:
        dep = t_now + i * ds
        if dep > t_now + wait_max + 1e-9:
            break
        if best is not None and dep >= best[0] - 1e-9:
            break
        limit_arr = best[0] if best is not None else dep + tof_max
        tof = tof_min
        while dep + tof < limit_arr - 1e-9:
            if util.compute_transfer(a, b, dep, tof) <= lim - MARGIN:
                best = (dep + tof, dep, tof)
                break
            tof += ts
        i += 1
    return best


def bits(m):
    while m:
        b = m & -m
        yield b.bit_length() - 1
        m ^= b


def main():
    rows, params = [], None
    with open(INSTANCE, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("c"):
                continue
            if line.startswith("p "):
                params = line.split()[2:]
            else:
                rows.append([float(x) for x in line.split()])
    orb = np.array(rows)
    N = len(orb)
    t0, t_max = float(params[0]), float(params[2])
    dv_max, dv_exc, E = float(params[3]), float(params[4]), int(float(params[5]))
    a_sma, ecc, inc, raan = orb[:, 0], orb[:, 1], orb[:, 2], orb[:, 3]
    low = a_sma < 8.0e6
    normals = np.stack([np.sin(inc) * np.sin(raan), -np.sin(inc) * np.cos(raan), np.cos(inc)], axis=1)
    H = normals * np.sqrt(MU * a_sma * (1.0 - ecc ** 2))[:, None]
    apo = a_sma * (1.0 + ecc)
    LB = np.linalg.norm(H[:, None, :] - H[None, :, :], axis=2) / np.maximum(apo[:, None], apo[None, :])
    np.fill_diagonal(LB, np.inf)

    easy_order = [np.argsort(np.where(LB[i] <= dv_max, LB[i], np.inf))[:int((LB[i] <= dv_max).sum())].tolist()
                  for i in range(N)]
    exc_order = [np.argsort(np.where(LB[i] <= dv_exc, LB[i], np.inf))[:int((LB[i] <= dv_exc).sum())].tolist()
                 for i in range(N)]
    se, pe, sx, px, ue = [0] * N, [0] * N, [0] * N, [0] * N, [0] * N
    for i in range(N):
        for j in easy_order[i]:
            se[i] |= 1 << j
            pe[j] |= 1 << i
            ue[i] |= 1 << j
            ue[j] |= 1 << i
        for j in exc_order[i]:
            sx[i] |= 1 << j
            px[j] |= 1 << i
    sa = [se[i] | sx[i] for i in range(N)]
    full = (1 << N) - 1

    parent = list(range(N))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(N):
        for j in easy_order[i]:
            parent[find(i)] = find(j)
    cnt = {}
    for i in range(N):
        cnt[find(i)] = cnt.get(find(i), 0) + 1
    big = max(cnt, key=cnt.get)
    small = [r for r in cnt if r != big]
    sigbit = [0] * N
    for i in range(N):
        if find(i) != big:
            sigbit[i] = 1 << small.index(find(i))

    def comps(mask):
        n, rem = 0, mask
        while rem:
            low_b = rem & -rem
            comp, fr = low_b, low_b
            while fr:
                b = fr & -fr
                fr ^= b
                new = ue[b.bit_length() - 1] & rem & ~comp
                comp |= new
                fr |= new
            rem &= ~comp
            n += 1
        return n

    high_nodes = sorted([i for i in range(N) if not low[i]], key=lambda i: inc[i])
    inc_sorted = [float(inc[i]) for i in high_nodes]
    prefix = [0]
    for i in high_nodes:
        prefix.append(prefix[-1] | (1 << i))
    highmask = prefix[-1]
    sweep_key = [SWEEP * float(inc[i]) for i in range(N)]
    allowed_starts = [s for s in range(N) if sweep_key[s] <= min(sweep_key) + START_TOL]

    def feasible(U, c, left):
        if DELTA > 0:
            vh = (full & ~U) & highmask
            if vh:
                incs = [float(inc[u]) for u in bits(vh)]
                l = bisect.bisect_right(inc_sorted, min(incs) + DELTA)
                r = bisect.bisect_left(inc_sorted, max(incs) - DELTA)
                if r > l and (prefix[r] & ~prefix[l]) & U:
                    return False
        if comps(U | (1 << c)) - 1 > left:
            return False
        avail = U | (1 << c)
        dead = 0
        for u in bits(U):
            if (pe[u] | (px[u] if left > 0 else 0)) & avail == 0:
                return False
            if sa[u] & U == 0:
                dead += 1
                if dead > 1:
                    return False
        # kontroll për komponente: çdo komponent i ndarë duhet të ketë hyrje dhe (përveç njërit) dalje
        cb = 1 << c
        rem = avail
        n_noexit = 0
        while rem:
            lb_ = rem & -rem
            comp, fr = lb_, lb_
            while fr:
                b = fr & -fr
                fr ^= b
                new = ue[b.bit_length() - 1] & rem & ~comp
                comp |= new
                fr |= new
            rem &= ~comp
            if comp & cb:
                continue
            outside = avail & ~comp
            can_in = can_out = False
            for u in bits(comp):
                if not can_in and (pe[u] | (px[u] if left > 0 else 0)) & outside:
                    can_in = True
                if not can_out and (se[u] | (sx[u] if left > 0 else 0)) & outside:
                    can_out = True
                if can_in and can_out:
                    break
            if not can_in:
                return False
            if not can_out:
                n_noexit += 1
                if n_noexit > 1:
                    return False
        return True

    n_proc = max(1, cpu_count() - 1)
    print(f"Instanca: {INSTANCE} | N = {N} | t_max = {t_max} | E = {E} | grupe të vogla: {len(small)}")
    print(f"Parametra: K për familje = {K_FAM}, pasues të lehtë = {K_EASY}, pasues përjashtimi = {M_EXC} | procese: {n_proc}")

    states = {}
    name = os.path.splitext(os.path.basename(INSTANCE))[0]
    if RESUME:
        with open(RESUME, "rb") as fh:
            states = pickle.load(fh)
        level_start = len(next(iter(states.values()))[1])
        print(f"Rinisje nga {RESUME}: {len(states)} gjendje me {level_start} domate të vizituara")
    else:
        for s in (allowed_starts if SWEEP else range(N)):
            states[(s, 1 << s, 0)] = (t0, (s,), (), (), (), sigbit[s])
        level_start = 1
    cache = {}
    start = time.time()
    with Pool(processes=n_proc, initializer=init_worker, initargs=(low.tolist(),)) as pool:
        for level in range(level_start, N):
            todo, wanted = [], {}
            for (c, vis, used), st in states.items():
                t = st[0]
                tk = int(round(t * 100))
                U = full & ~vis
                front = max((sweep_key[u] for u in st[1] if not low[u]), default=-1e9)

                def okj(j, front=front):
                    return (not SWEEP) or bool(low[j]) or sweep_key[j] >= front - BACK
                cl = []
                for j in easy_order[c]:
                    if (U >> j) & 1 and okj(j):
                        cl.append((j, 0))
                        if len(cl) >= K_EASY:
                            break
                if used < E:
                    n_v = 0
                    for j in easy_order[c]:          # variante përjashtimi të kalimeve të lehta
                        if (U >> j) & 1 and okj(j):
                            cl.append((j, 1))
                            n_v += 1
                            if n_v >= K_EASY:
                                break
                    n_cross = n_same = 0
                    for j in exc_order[c]:           # kalime strukturore, të ndara sipas llojit
                        if (U >> j) & 1 and LB[c, j] > dv_max and okj(j):
                            if bool(low[j]) != bool(low[c]):
                                if n_cross < M_EXC:
                                    cl.append((j, 1))
                                    n_cross += 1
                            elif n_same < 3:
                                cl.append((j, 1))
                                n_same += 1
                            if n_cross >= M_EXC and n_same >= 3:
                                break
                wanted[(c, vis, used)] = cl
                for j, kd in cl:
                    key = (c, j, tk, kd)
                    if key not in cache and key not in wanted:
                        cache[key] = None
                        todo.append((key, (c, j, t, dv_exc if kd else dv_max, kd)))
            if todo:
                res = pool.map(eval_hop, [x[1] for x in todo], chunksize=max(1, len(todo) // (n_proc * 4)))
                for (key, _), r in zip(todo, res):
                    cache[key] = r

            children = {}
            for (c, vis, used), st in states.items():
                t, path, kinds, deps, tofs, sig = st
                tk = int(round(t * 100))
                for j, kd in wanted[(c, vis, used)]:
                    r = cache[(c, j, tk, kd)]
                    if r is None or r[0] > t_max:
                        continue
                    key2 = (j, vis | (1 << j), used + kd)
                    if key2 not in children or r[0] < children[key2][0]:
                        children[key2] = (r[0], path + (j,), kinds + (kd,), deps + (r[1],), tofs + (r[2],), sig | sigbit[j])

            groups = {}
            for key2, v in children.items():
                j2, vis2_, used2_ = key2
                U2 = full & ~vis2_
                near = U2 | (1 << j2)
                danger = 0
                for u in bits(U2):
                    if (ue[u] & near).bit_count() <= 1:
                        danger += 1
                groups.setdefault((v[5], key2[2]), []).append((v[0] + ALPHA * danger, key2, v))
            prev = states
            states = {}
            for fam, lst in groups.items():
                lst.sort(key=lambda z: z[0])
                kept = 0
                for t2, key2, v in lst:
                    j, vis2, used2 = key2
                    if feasible(full & ~vis2, j, E - used2):
                        states[key2] = v
                        kept += 1
                        if kept >= K_FAM:
                            break
            if not states:
                print(f"  Niveli {level + 1}: asnjë gjendje nuk mbijetoi. Beam-i vdiq këtu.")
                with open(f"lazy_ckpt_{name}_death.pkl", "wb") as fh:
                    pickle.dump(prev, fh)
                print(f"  Gjendjet para vdekjes u ruajtën në lazy_ckpt_{name}_death.pkl")
                kmin = min(prev, key=lambda k: prev[k][0])
                unv = [u for u in range(N) if not (kmin[1] >> u) & 1]
                print(f"  Gjendja më e mirë para vdekjes: {N - len(unv)} domate të vizituara, koha {prev[kmin][0]:.1f}, "
                      f"përjashtime {kmin[2]}")
                print(f"  Të pavizituara ({len(unv)}):", unv[:80])
                return
            if level % 10 == 0:
                with open(f"lazy_ckpt_{name}_{level + 1}.pkl", "wb") as fh:
                    pickle.dump(states, fh)
            if level % 10 == 0 or level == N - 1:
                ts = [v[0] for v in states.values()]
                print(f"  Niveli {level + 1:4d}/{N}: {len(states):4d} gjendje | koha më e mirë = {min(ts):8.2f} | "
                      f"{time.time() - start:6.0f} s | kalime të llogaritura: {len(cache)}")

    best = min(states.values(), key=lambda v: v[0])
    t, path, kinds, deps, tofs, _ = best
    x = list(deps) + list(tofs) + list(path)
    print(f"\nf më e mirë = {t:.3f} ditë | përjashtime: {sum(kinds)} | koha totale: {time.time() - start:.0f} s")
    ok, fv, _, _ = validator.validate(x, TomatoProblem(INSTANCE), show_hops=False)
    if ok:
        out = f"best_lazy_{name}_f{fv:.3f}.json"
        json.dump({"x": [float(v) for v in x[:2 * (N - 1)]] + [int(v) for v in path], "f": fv}, open(out, "w"))
        print("U ruajt:", out)


if __name__ == "__main__":
    main()