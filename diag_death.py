import pickle
import sys

INSTANCE = sys.argv[1] if len(sys.argv) > 1 else "problems/hard.kttsp"
PKL = sys.argv[2] if len(sys.argv) > 2 else "lazy_ckpt_hard_death.pkl"
BACK = float(sys.argv[3]) if len(sys.argv) > 3 else 0.06


def main():
    rows = []
    with open(INSTANCE, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line and not line.startswith("c") and not line.startswith("p "):
                rows.append([float(x) for x in line.split()])
    N = len(rows)
    inc = [r[2] for r in rows]
    low = [r[0] < 8.0e6 for r in rows]
    with open(PKL, "rb") as fh:
        states = pickle.load(fh)
    (c, vis, used), st = min(states.items(), key=lambda kv: kv[1][0])
    t, path, kinds = st[0], st[1], st[2]
    unv = [u for u in range(N) if not (vis >> u) & 1]
    vh = [u for u in path if not low[u]]
    vl = [u for u in path if low[u]]
    print(f"Gjendja më e mirë: {len(path)} të vizituara, koha {t:.1f}, përjashtime {used}")
    print(f"Të vizituara: {len(vh)} të larta, {len(vl)} të ulëta")
    front = max(inc[u] for u in vh)
    print(f"Domatja aktuale: {c} ({'ulët' if low[c] else 'lartë'}, inklinim {inc[c]:.3f}) | "
          f"fronti i të larta = {front:.3f} | e para e të larta: inklinim {inc[vh[0]]:.3f}")
    print("\nPërjashtimet e përdorura (nga -> te, inklinimet):")
    for k, kd in enumerate(kinds):
        if kd:
            a, b = path[k], path[k + 1]
            print(f"  pas {k + 1} domatesh: {a} ({'ulët' if low[a] else 'lartë'}, {inc[a]:.3f}) -> "
                  f"{b} ({'ulët' if low[b] else 'lartë'}, {inc[b]:.3f})")
    uh = [u for u in unv if not low[u]]
    ul = [u for u in unv if low[u]]
    behind = [u for u in uh if inc[u] < front - BACK]
    print(f"\nTë pavizituara: {len(uh)} të larta, {len(ul)} të ulëta")
    print(f"  të larta TEK mbrapa frontit (më shumë se {BACK} nën të): {len(behind)}")
    if behind:
        print(f"  inklinimet e tyre: nga {min(inc[u] for u in behind):.3f} në {max(inc[u] for u in behind):.3f}")
    print("\nHistogram (inklinim, kutitë 0.25): të pavizituara të larta / të gjitha të larta | të pavizituara të ulëta / të gjitha të ulëta")
    for b0 in [x * 0.25 for x in range(0, 13)]:
        tot_h = sum(1 for u in range(N) if not low[u] and b0 <= inc[u] < b0 + 0.25)
        un_h = sum(1 for u in uh if b0 <= inc[u] < b0 + 0.25)
        tot_l = sum(1 for u in range(N) if low[u] and b0 <= inc[u] < b0 + 0.25)
        un_l = sum(1 for u in ul if b0 <= inc[u] < b0 + 0.25)
        if tot_h or tot_l:
            print(f"  {b0:4.2f}-{b0 + 0.25:4.2f}:  {un_h:4d}/{tot_h:4d}  |  {un_l:4d}/{tot_l:4d}")
    print("\n10 domatet e fundit të rrugës (indeks, lloji, inklinim):")
    print("  " + " -> ".join(f"{u}({'L' if low[u] else 'H'},{inc[u]:.3f})" for u in path[-10:]))


if __name__ == "__main__":
    main()