import numpy as np

try:
    from utils_users import MU_MOON as MU
except Exception:
    MU = 4.9048695e12

# 1) Lexo instancën
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

a = orb[:, 0]
e = orb[:, 1]
inc = orb[:, 2]
raan = orb[:, 3]
low = a < 8.0e6

# 2) Vektori i momentit këndor h (drejtimi = normalja e planit)
normals = np.stack([np.sin(inc) * np.sin(raan),
                    -np.sin(inc) * np.cos(raan),
                    np.cos(inc)], axis=1)
h_mag = np.sqrt(MU * a * (1.0 - e ** 2))
H = normals * h_mag[:, None]
apo = a * (1.0 + e)

# 3) Kufiri i poshtëm i dV për çdo çift: |h_j - h_i| / r_max
diff = np.linalg.norm(H[:, None, :] - H[None, :, :], axis=2)
rmax = np.maximum(apo[:, None], apo[None, :])
LB = diff / rmax
np.fill_diagonal(LB, 0.0)

# 4) Kontroll: asnjë dV e gjetur nuk duhet të jetë nën kufirin e poshtëm
eye = np.eye(N, dtype=bool)
viol = (best < LB * 0.995) & ~eye
print("Kontroll i kufirit të poshtëm: çifte ku dV e gjetur < kufirin =", int(viol.sum()),
      "(duhet të jetë 0)")
ratio = (best[~eye] / np.maximum(LB[~eye], 1e-9))
print("Raporti dV_gjetur / kufiri: min =", round(float(ratio.min()), 3),
      " mediane =", round(float(np.median(ratio)), 3))
print()

# 5) Komponentet e grafit "mundësisht i lehtë" (LB <= dV_max)
possible = (LB <= dv_max) & ~eye
print("Çifte të renditura që MUND të jenë të lehta (LB <= dV_max):", int(possible.sum()),
      "| të gjetura të lehta në rrjetën e përafërt:", int(((best <= dv_max) & ~eye).sum()))

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

comps = components(possible)
comps.sort(key=len, reverse=True)
print()
print("Komponente në grafin më optimist të kalimeve të lehta:", len(comps))
for k, c in enumerate(comps):
    tag = "ulët" if all(low[i] for i in c) else ("lartë" if not any(low[i] for i in c) else "përzier")
    print(f"  K{k}: madhësia {len(c):2d} ({tag}): {c}")
print()
print(f"Kusht i nevojshëm: komponentet <= E + 1 = {E + 1}  ->",
      "PLOTËSOHET" if len(comps) <= E + 1 else "NUK PLOTËSOHET")
print()

# 6) Kufiri i poshtëm mes komponenteve (vlerat <= dV_exception; '.' = jashtë kufirit)
K = len(comps)
print("Kufiri i poshtëm minimal i dV mes komponenteve (vetëm vlerat <= dV_exception, përndryshe '.'):")
header = "      " + " ".join(f"K{k:<4d}" for k in range(K))
print(header)
for p in range(K):
    cells = []
    for q in range(K):
        if p == q:
            cells.append("  -  ")
        else:
            m = min(LB[i, j] for i in comps[p] for j in comps[q])
            cells.append(f"{m:5.0f}" if m <= dv_exc else "  .  ")
    print(f"K{p:<4d} " + " ".join(cells))