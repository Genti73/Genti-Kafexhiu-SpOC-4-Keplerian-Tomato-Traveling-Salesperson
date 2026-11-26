import json
import sys
import time
from utils_users import TomatoProblem
import validator

INSTANCE = sys.argv[3] if len(sys.argv) > 3 else "problems/easy.kttsp"
SRC = sys.argv[1] if len(sys.argv) > 1 else "best_solution.json"
STEP = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
MARGIN = 0.02


def main():
    u = TomatoProblem(INSTANCE)
    N = u.n_tomatoes
    x = json.load(open(SRC))["x"]
    t_old, tof_old = x[:N - 1], x[N - 1:2 * (N - 1)]
    pi = [int(v) for v in x[2 * (N - 1):]]
    print(f"Burimi: {SRC} | hapi i rrjetës: {STEP} ditë")

    try:
        t_now = float(u.t0.mjd2000)
    except Exception:
        t_now = 0.0
    t_new, tof_new = [], []
    start = time.time()
    for k in range(N - 1):
        a, b = pi[k], pi[k + 1]
        dv_old = u.compute_transfer(a, b, t_old[k], tof_old[k])
        lim = u.dv_threshold if dv_old <= u.dv_threshold else u.dv_exception
        bound = t_old[k] + tof_old[k]                  # mbërritja e vjetër (gjithmonë e mundshme)
        best = (bound, t_old[k], tof_old[k])
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
        gain = bound - best[0]
        t_new.append(best[1])
        tof_new.append(best[2])
        t_now = best[1] + best[2]
        if (k + 1) % 8 == 0 or k == N - 2:
            print(f"  kalimi {k + 1}/{N - 1}: koha aktuale = {t_now:.3f} (kursimi i akumuluar: {gain:.3f}) "
                  f"| {time.time() - start:.0f} s")

    x_new = t_new + tof_new + pi
    f_old = t_old[-1] + tof_old[-1]
    print(f"\nf e vjetër = {f_old:.4f} | f e re = {t_now:.4f} | kursim = {f_old - t_now:.4f} ditë\n")
    ok, f, _, _ = validator.validate(x_new, u, show_hops=False)
    if ok:
        out = f"best_polished_f{f:.3f}.json"
        json.dump({"x": [float(v) for v in x_new[:2 * (N - 1)]] + pi, "f": f}, open(out, "w"))
        print("U ruajt:", out)


if __name__ == "__main__":
    main()