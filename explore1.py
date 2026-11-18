import time
import numpy as np
from utils_users import TomatoProblem

util = TomatoProblem("problems/easy.kttsp")
N = util.n_tomatoes
print("Numri i domateve N =", N)
print("t_max =", util.max_time, "| dV_max =", util.dv_threshold,
      "| dV_exception =", util.dv_exception, "| E =", util.n_exceptions)

a_km = util.tomato_orbital_parameters[:, 0] / 1000.0
print("Semi-major axis (km): min =", round(a_km.min()), " max =", round(a_km.max()))
print("Domate me a < 8000 km:", [i for i in range(N) if a_km[i] < 8000])

t_grid = np.arange(0.0, 60.0, 1.5)
tof_grid = np.array([0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])

best = np.full((N, N), np.inf)
start = time.time()
for i in range(N):
    for j in range(N):
        if i == j:
            continue
        for t in t_grid:
            for tof in tof_grid:
                dv = util.compute_transfer(i, j, t, tof)
                if dv < best[i, j]:
                    best[i, j] = dv
    print(f"rreshti {i + 1}/{N} gati ({time.time() - start:.0f} s)")

np.save("min_dv_easy.npy", best)

mask = ~np.eye(N, dtype=bool)
n_easy = np.sum((best <= util.dv_threshold) & mask)
n_exc = np.sum((best > util.dv_threshold) & (best <= util.dv_exception) & mask)
n_none = np.sum((best > util.dv_exception) & mask)
print()
print("Çifte me dV <= dV_max (lehtë):      ", n_easy)
print("Çifte me dV_max < dV <= përjashtim: ", n_exc)
print("Çifte mbi përjashtim (të vështira): ", n_none)
print("Çifte gjithsej:                     ", N * (N - 1))
print()
print("Numri i pasuesve të lehtë për çdo domate:")
print([int(np.sum((best[i] <= util.dv_threshold) & mask[i])) for i in range(N)])