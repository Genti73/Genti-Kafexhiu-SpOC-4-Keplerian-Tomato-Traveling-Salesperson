import numpy as np

# 1) Lexo instancën direkt nga skedari
rows = []
params = None
with open("problems/easy.kttsp", "r", encoding="utf-8") as f:
    for raw in f:
        line = raw.strip()
        if not line or line.startswith("c"):
            continue
        if line.startswith("p "):
            params = line.split()[2:]
            continue
        rows.append([float(x) for x in line.split()])

orb = np.array(rows)
N = len(orb)
dv_max = float(params[3])
dv_exc = float(params[4])
E = int(float(params[5]))
best = np.load("min_dv_easy.npy")

eye = np.eye(N, dtype=bool)
easy = (best <= dv_max) & ~eye
exc = (best <= dv_exc) & ~eye

# 2) Grupet sipas lartësisë
a_km = orb[:, 0] / 1000.0
low = a_km < 8000
low_ids = [i for i in range(N) if low[i]]
print("N =", N, "| E =", E)
print("Domate në orbitë të ulët:", low_ids)
print()

# 3) Këndi mes planeve orbitale
inc = orb[:, 2]
raan = orb[:, 3]
normals = np.stack([np.sin(inc) * np.sin(raan),
                    -np.sin(inc) * np.cos(raan),
                    np.cos(inc)], axis=1)
cosang = np.clip(normals @ normals.T, -1.0, 1.0)
ang = np.degrees(np.arccos(cosang))

# 4) Lidhja mes këndit të planeve dhe dV (vetëm çifte brenda grupit të lartë)
high_ids = [i for i in range(N) if not low[i]]
print("Çifte brenda grupit të lartë: kënd plani -> dV minimal i gjetur")
bins = [0, 5, 10, 20, 40, 90, 181]
for lo, hi in zip(bins[:-1], bins[1:]):
    vals = [best[i, j] for i in high_ids for j in high_ids
            if i != j and lo <= ang[i, j] < hi]
    if vals:
        print(f"  {lo:3d}-{hi:3d} gradë: {len(vals):4d} çifte, "
              f"dV mediane = {np.median(vals):8.1f}, minimale = {np.min(vals):8.1f}")
print()

# 5) Lidhjet mes dy grupeve
cross = [(i, j) for i in range(N) for j in range(N)
         if i != j and low[i] != low[j]]
print("Çifte mes grupit të ulët dhe të lartë:", len(cross))
print("  me dV <= dV_max:      ", sum(1 for i, j in cross if easy[i, j]))
print("  me dV <= përjashtim:  ", sum(1 for i, j in cross if exc[i, j]))
print("  dV minimale e gjetur: ", round(min(best[i, j] for i, j in cross), 1))
print()

# 6) Pasues dhe paraardhës të lehtë
out_deg = easy.sum(axis=1)
in_deg = easy.sum(axis=0)
print("Pasues të lehtë (out):  ", [int(x) for x in out_deg])
print("Paraardhës të lehtë (in):", [int(x) for x in in_deg])
print("Pa asnjë pasues të lehtë:   ", [i for i in range(N) if out_deg[i] == 0])
print("Pa asnjë paraardhës të lehtë:", [i for i in range(N) if in_deg[i] == 0])
print()

# 7) Komponentet e lidhura të grafit të lehtë (pa drejtim)
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

comps = components(easy | easy.T)
comps.sort(key=len, reverse=True)
print("Komponente të lidhura të grafit të lehtë:", len(comps))
for c in comps:
    print(f"  madhësia {len(c):2d}: {c}")
print()
print(f"Kusht i nevojshëm: komponentet <= E + 1 = {E + 1}  ->",
      "PLOTËSOHET" if len(comps) <= E + 1 else "NUK PLOTËSOHET (me këtë rrjetë të përafërt)")