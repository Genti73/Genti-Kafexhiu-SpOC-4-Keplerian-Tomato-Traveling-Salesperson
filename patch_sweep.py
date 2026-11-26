with open("lazy_beam.py", "r", encoding="utf-8") as f:
    src = f.read()


def replace_once(s, old, new, label):
    if s.count(old) != 1:
        print(f"STOP: blloku '{label}' nuk u gjet saktësisht një herë. Asgjë nuk u ndryshua.")
        raise SystemExit(1)
    return s.replace(old, new)


# 1) importet dhe parametrat
src = replace_once(src, "import json\nimport os\nimport pickle\nimport sys\n",
                   "import bisect\nimport json\nimport os\nimport pickle\nimport sys\n", "import bisect")
src = replace_once(
    src,
    "RESUME = sys.argv[6] if len(sys.argv) > 6 else None\n",
    "RESUME = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6] != \"-\" else None\n"
    "DELTA = float(sys.argv[7]) if len(sys.argv) > 7 else 0.04\n"
    "SWEEP = int(float(sys.argv[8])) if len(sys.argv) > 8 else 0\n"
    "BACK = float(sys.argv[9]) if len(sys.argv) > 9 else 0.06\n"
    "START_TOL = float(sys.argv[10]) if len(sys.argv) > 10 else 0.08\n",
    "parametrat")

# 2) rregulli i intervalit dhe çelësat e skanimit
src = replace_once(
    src,
    "    def feasible(U, c, left):\n        if comps(U | (1 << c)) - 1 > left:\n            return False\n",
    """    high_nodes = sorted([i for i in range(N) if not low[i]], key=lambda i: inc[i])
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
""",
    "rregulli i intervalit")

# 3) gjendjet fillestare
src = replace_once(
    src,
    "        for s in range(N):\n            states[(s, 1 << s, 0)] = (t0, (s,), (), (), (), sigbit[s])\n",
    "        for s in (allowed_starts if SWEEP else range(N)):\n            states[(s, 1 << s, 0)] = (t0, (s,), (), (), (), sigbit[s])\n",
    "gjendjet fillestare")

# 4) filtri i drejtimit te kandidatët
src = replace_once(
    src,
    "                U = full & ~vis\n                cl = []\n",
    """                U = full & ~vis
                front = max((sweep_key[u] for u in st[1] if not low[u]), default=-1e9)

                def okj(j, front=front):
                    return (not SWEEP) or bool(low[j]) or sweep_key[j] >= front - BACK
                cl = []
""",
    "fillimi i ciklit të kandidatëve")
src = replace_once(src, "                    if (U >> j) & 1:\n                        cl.append((j, 0))\n",
                   "                    if (U >> j) & 1 and okj(j):\n                        cl.append((j, 0))\n", "kandidatët e lehtë")
src = replace_once(src, "                        if (U >> j) & 1:\n                            cl.append((j, 1))\n                            n_v += 1\n",
                   "                        if (U >> j) & 1 and okj(j):\n                            cl.append((j, 1))\n                            n_v += 1\n",
                   "variantet e përjashtimit")

# 5) kandidatët strukturorë të ndarë sipas llojit
src = replace_once(
    src,
    """                    n_ex = 0
                    for j in exc_order[c]:           # kalime strukturore (kufiri teorik mbi dV_max)
                        if (U >> j) & 1 and LB[c, j] > dv_max:
                            cl.append((j, 1))
                            n_ex += 1
                            if n_ex >= M_EXC:
                                break
""",
    """                    n_cross = n_same = 0
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
""",
    "kandidatët strukturorë")

with open("lazy_beam.py", "w", encoding="utf-8") as f:
    f.write(src)
print("OK: lazy_beam.py u përditësua (rregull intervali, skanim me drejtim, kandidatë të ndarë).")