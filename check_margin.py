import json
import sys
from utils_users import TomatoProblem

path = sys.argv[1] if len(sys.argv) > 1 else "best_solution.json"
x = json.load(open(path))["x"]
u = TomatoProblem("problems/easy.kttsp")
N = u.n_tomatoes
t, tof, pi = x[:N - 1], x[N - 1:2 * (N - 1)], x[2 * (N - 1):]
rows = []
for k in range(N - 1):
    dv = u.compute_transfer(int(pi[k]), int(pi[k + 1]), t[k], tof[k])
    lim = 100.0 if dv <= 100.0 else 600.0
    rows.append((lim - dv, k + 1, pi[k], pi[k + 1], dv, lim))
rows.sort()
print("10 kalimet me diferencën më të vogël ndaj kufirit:")
for m, k, a, b, dv, lim in rows[:10]:
    print(f"  #{k:2d} {a:2d}->{b:2d}  dV = {dv:.6f}  kufiri = {lim:.0f}  diferenca = {m:.6f}")