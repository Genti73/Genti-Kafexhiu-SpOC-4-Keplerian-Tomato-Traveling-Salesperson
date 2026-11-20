import numpy as np

STRONG = 0.20
INSTANCE = "problems/easy.kttsp"


def load(path):
    d = np.load(path)
    rows = {}
    for (i, j), idx in zip(d["pairs"], d["tof_idx"]):
        if (idx >= 0).any():
            rows[(int(i), int(j))] = idx
    return rows, d["T"]


def windows(idx, T):
    ok = idx >= 0
    out, k, n = [], 0, len(ok)
    while k < n:
        if ok[k]:
            k0 = k
            while k + 1 < n and ok[k + 1]:
                k += 1
            out.append((float(T[k0]), float(T[k])))
        k += 1
    return out


def fmt(w, m=4):
    s = ", ".join(f"{a:.1f}-{b:.1f}" for a, b in w[:m])
    return s + (f" (+{len(w) - m})" if len(w) > m else "")


def main():
    a = []
    with open(INSTANCE, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line and not line.startswith("c") and not line.startswith("p "):
                a.append(float(line.split()[0]))
    N = len(a)
    low = [x < 8.0e6 for x in a]
    easy, Te = load("edge_table_easy.npz")
    exc, Tx = load("edge_table_exc.npz")
    av = {p: float((idx >= 0).mean()) for p, idx in easy.items()}

    parent = list(range(N))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (i, j), v in av.items():
        if v >= STRONG:
            parent[find(i)] = find(j)
    groups = {}
    for i in range(N):
        groups.setdefault(find(i), []).append(i)
    comps = sorted(groups.values(), key=len, reverse=True)
    cid = {n: k for k, c in enumerate(comps) for n in c}
    print(f"Grupe të forta (lidhje të hapura >= {STRONG:.0%} e kohës): {len(comps)}")
    for k, c in enumerate(comps):
        print(f"  G{k} ({'ulët' if low[c[0]] else 'lartë'}, {len(c)}): {sorted(c)}")

    weak = sorted(((v, p) for p, v in av.items() if v < STRONG))
    print(f"\nLidhje të lehta të dobëta (< {STRONG:.0%}): {len(weak)}")
    for v, (i, j) in weak:
        tag = "brenda" if cid[i] == cid[j] else f"G{cid[i]}->G{cid[j]}"
        print(f"  {i:2d}->{j:2d} [{tag}] {v * 100:5.1f}% | dritare (ditë): {fmt(windows(easy[(i, j)], Te))}")

    print("\nPërjashtime (ulët<->lartë): për çdo grup të ulët, grupet e larta ku lidhet (çifte | disponueshmëria maks)")
    for k, c in enumerate(comps):
        if not low[c[0]]:
            continue
        summ = {}
        for (i, j), idx in exc.items():
            if (i in c) != (j in c) and (i in c or j in c):
                o = j if i in c else i
                kk = cid[o]
                n, m = summ.get(kk, (0, 0.0))
                summ[kk] = (n + 1, max(m, float((idx >= 0).mean())))
        txt = ", ".join(f"G{kk}: {n} | {m * 100:.0f}%" for kk, (n, m) in sorted(summ.items()))
        print(f"  G{k} {sorted(c)} -> {txt}")


if __name__ == "__main__":
    main()