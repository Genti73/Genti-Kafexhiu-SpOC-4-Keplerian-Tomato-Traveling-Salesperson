import json
import sys
import numpy as np
from utils_users import TomatoProblem

TOL = 1e-9


def _fail(errors, warnings):
    print("REZULTATI: ZGJIDHJA NUK ËSHTË E VLEFSHME")
    for e in errors:
        print("  GABIM:", e)
    return False, None, errors, warnings


def validate(x, util, show_hops=True):
    """Kontrollon një zgjidhje x = [t_1..t_{N-1}, tof_1..tof_{N-1}, pi_1..pi_N].
    Kthen (ok, f, gabime, paralajmërime)."""
    N = util.n_tomatoes
    errors, warnings = [], []
    x = list(x)

    if len(x) != 3 * N - 2:
        errors.append(f"Gjatësia e x është {len(x)}, duhet {3 * N - 2}")
        return _fail(errors, warnings)

    t = np.array(x[:N - 1], dtype=float)
    tof = np.array(x[N - 1:2 * (N - 1)], dtype=float)
    pi_raw = np.array(x[2 * (N - 1):], dtype=float)

    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(tof)) and np.all(np.isfinite(pi_raw))):
        errors.append("x përmban vlera jo-numerike (NaN ose inf)")
        return _fail(errors, warnings)
    if np.any(np.abs(pi_raw - np.round(pi_raw)) > 1e-9):
        errors.append("Indekset e domateve nuk janë numra të plotë")
        return _fail(errors, warnings)
    pi = [int(round(v)) for v in pi_raw]

    if sorted(pi) != list(range(N)):
        missing = sorted(set(range(N)) - set(pi))
        dup = sorted({v for v in pi if pi.count(v) > 1})
        errors.append(f"Nuk është permutacion i vlefshëm. Mungojnë: {missing} | Të përsëritura: {dup}")
        return _fail(errors, warnings)

    try:
        t0 = float(util.t0.mjd2000)
    except Exception:
        t0 = 0.0

    if t[0] < t0 - TOL:
        errors.append(f"Nisja e parë ({t[0]}) është para kohës fillestare t0 = {t0}")

    bad_tof = [k + 1 for k in range(N - 1) if tof[k] < util.min_tof - TOL]
    if bad_tof:
        errors.append(f"Koha e fluturimit nën minimumin {util.min_tof} ditë në transferet: {bad_tof}")

    bad_order = [k + 1 for k in range(N - 2) if t[k] + tof[k] > t[k + 1] + TOL]
    if bad_order:
        errors.append(f"Nisja para mbërritjes (t_i + tof_i > t_(i+1)) në transferet: {bad_order}")

    dv = np.array([util.compute_transfer(pi[k], pi[k + 1], float(t[k]), float(tof[k]))
                   for k in range(N - 1)], dtype=float)
    dv_max, dv_exc, E = util.dv_threshold, util.dv_exception, util.n_exceptions
    is_exc = (dv > dv_max) & (dv <= dv_exc)
    is_bad = ~(dv <= dv_exc)          # përfshin edhe inf/nan
    n_exc, n_bad = int(is_exc.sum()), int(is_bad.sum())

    if n_bad:
        errors.append(f"{n_bad} transfere kalojnë dV_exception = {dv_exc}: transferet "
                      f"{[k + 1 for k in range(N - 1) if is_bad[k]]}")
    if n_exc > E:
        errors.append(f"{n_exc} përjashtime të përdorura, lejohen vetëm {E}: transferet "
                      f"{[k + 1 for k in range(N - 1) if is_exc[k]]}")

    margin = 0.5
    close = [k + 1 for k in range(N - 1)
             if np.isfinite(dv[k]) and (abs(dv[k] - dv_max) < margin or abs(dv[k] - dv_exc) < margin)]
    if close:
        warnings.append(f"dV shumë afër kufirit (nën {margin} m/s) në transferet: {close}")

    f = float(t[-1] + tof[-1])
    if f > util.max_time + TOL:
        errors.append(f"Koha totale f = {f:.4f} kalon t_max = {util.max_time}")

    ok = len(errors) == 0

    if show_hops:
        print(f"{'#':>3} {'nga':>4} {'te':>4} {'t_nisjes':>9} {'tof':>7} {'mbërritja':>10} {'dV':>10}  lloji")
        for k in range(N - 1):
            kind = "lehtë" if dv[k] <= dv_max else ("PËRJASHTIM" if dv[k] <= dv_exc else "E PAVLEFSHME")
            print(f"{k + 1:>3} {pi[k]:>4} {pi[k + 1]:>4} {t[k]:>9.3f} {tof[k]:>7.3f} "
                  f"{t[k] + tof[k]:>10.3f} {dv[k]:>10.2f}  {kind}")
        print()

    print(f"Tomate: {N} | transfere: {N - 1} | të lehta: {N - 1 - n_exc - n_bad} | "
          f"përjashtime: {n_exc}/{E} | të pavlefshme: {n_bad}")
    print(f"Koha totale f = {f:.4f} ditë (kufiri {util.max_time})")
    for w in warnings:
        print("PARALAJMËRIM:", w)
    if ok:
        print("REZULTATI: ZGJIDHJA ËSHTË E VLEFSHME")
    else:
        print("REZULTATI: ZGJIDHJA NUK ËSHTË E VLEFSHME")
        for e in errors:
            print("  GABIM:", e)
    return ok, f, errors, warnings


def load_x(path):
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        data = data["x"]
    return data


def demo_solution(N):
    """Zgjidhje naive vetëm për të provuar validuesin: domatet 0,1,2,... me nga 4 ditë mes nisjeve."""
    t = [4.0 * k for k in range(N - 1)]
    tof = [1.0] * (N - 1)
    pi = list(range(N))
    return t + tof + pi


def main():
    if len(sys.argv) < 2:
        print("Përdorimi:  python validator.py demo   |   python validator.py zgjidhja.json [instanca.kttsp]")
        return
    if sys.argv[1] == "demo":
        instance = sys.argv[2] if len(sys.argv) > 2 else "problems/easy.kttsp"
        util = TomatoProblem(instance)
        validate(demo_solution(util.n_tomatoes), util)
    else:
        instance = sys.argv[2] if len(sys.argv) > 2 else "problems/easy.kttsp"
        util = TomatoProblem(instance)
        validate(load_x(sys.argv[1]), util)


if __name__ == "__main__":
    main()