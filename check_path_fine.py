import json
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "best_solution.json"
sys.argv = sys.argv[:1]          # solver.py e lexon argumentin e parë si numër

from utils_users import TomatoProblem
from solver import Table, load_instance, INSTANCE


def trace(path, kinds, easy, exc, t0):
    t_now = t0
    for k, (a, b, kd) in enumerate(zip(path, path[1:], kinds)):
        r = (exc if kd else easy).hop(a, b, t_now)
        if r is None:
            return None, (k + 1, a, b, kd, t_now)
        dep, tof = r
        t_now = max(dep, t_now) + tof
    return t_now, None


def main():
    N, t0, t_max, E = load_instance()
    u = TomatoProblem(INSTANCE)
    x = json.load(open(SRC))["x"]
    t, tof = x[:N - 1], x[N - 1:2 * (N - 1)]
    pi = [int(v) for v in x[2 * (N - 1):]]
    kinds = [0 if u.compute_transfer(pi[k], pi[k + 1], t[k], tof[k]) <= u.dv_threshold else 1
             for k in range(N - 1)]
    print(f"Zgjidhja: {SRC} | f reale = {t[-1] + tof[-1]:.3f} | përjashtime: {sum(kinds)}")
    print("Renditja:", pi)
    for name, fe, fx in (("E TRASHË (hapi 0.05)", "edge_table_easy_coarse.npz", "edge_table_exc_coarse.npz"),
                         ("E IMËT (hapi 0.02)", "edge_table_easy_fine.npz", "edge_table_exc_fine.npz")):
        f, fail = trace(pi, kinds, Table(fe), Table(fx), t0)
        if f is not None:
            print(f"  {name}: f = {f:.3f}")
        else:
            k, a, b, kd, tn = fail
            print(f"  {name}: DËSHTON te kalimi #{k} ({a}->{b}, {'përjashtim' if kd else 'i lehtë'}) "
                  f"kur koha aktuale = {tn:.2f}")


if __name__ == "__main__":
    main()