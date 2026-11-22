import sys
import numpy as np

PATH = sys.argv[1] if len(sys.argv) > 1 else "problems/medium.kttsp"
MU = 4.9048695e12


def main():
    rows, params = [], None
    with open(PATH, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("c"):
                continue
            if line.startswith("p "):
                params = line.split()[2:]
            else:
                rows.append([float(x) for x in line.split()])
    orb = np.array(rows)
    N = len(orb)
    t0, min_tof, t_max, dv_max, dv_exc, E = (float(params[0]), float(params[1]), float(params[2]),
                                             float(params[3]), float(params[4]), int(float(params[5])))
    print(f"Instanca: {PATH}")
    print(f"N = {N} | t0 = {t0} | tof_min = {min_tof} | t_max = {t_max} | dV_max = {dv_max} | "
          f"dV_përjashtim = {dv_exc} | E = {E}")

    a, e, inc, raan = orb[:, 0], orb[:, 1], orb[:, 2], orb[:, 3]
    order = np.argsort(a)
    bands, cur = [], [order[0]]
    for k in order[1:]:
        if a[k] > a[cur[-1]] * 1.05:
            bands.append(cur)
            cur = [k]
        else:
            cur.append(k)
    bands.append(cur)
    print(f"\nBanda lartësish (boshllëk > 5% në gjysmëboshtin kryesor): {len(bands)}")
    for b in bands:
        print(f"  {len(b):4d} domate | a nga {a[b].min() / 1000:8.0f} në {a[b].max() / 1000:8.0f} km")

    normals = np.stack([np.sin(inc) * np.sin(raan), -np.sin(inc) * np.cos(raan), np.cos(inc)], axis=1)
    H = normals * np.sqrt(MU * a * (1.0 - e ** 2))[:, None]
    apo = a * (1.0 + e)
    LB = np.linalg.norm(H[:, None, :] - H[None, :, :], axis=2) / np.maximum(apo[:, None], apo[None, :])
    np.fill_diagonal(LB, np.inf)

    easy = LB <= dv_max
    exc = (LB <= dv_exc) & ~easy
    n_easy, n_exc = int(easy.sum()), int(exc.sum())
    outd = easy.sum(axis=1)
    ind = easy.sum(axis=0)
    print(f"\nÇifte të renditura kandidate të lehta (LB <= dV_max): {n_easy} "
          f"({n_easy / (N * (N - 1)) * 100:.2f}% e të gjithave)")
    print(f"Pasues kandidatë për domate: min = {outd.min()}, mediane = {int(np.median(outd))}, maksimum = {outd.max()}")
    print(f"Domate pa asnjë pasues: {int((outd == 0).sum())} | pa asnjë paraardhës: {int((ind == 0).sum())}")
    print(f"Çifte të mundshme si përjashtim (dV_max < LB <= dV_përjashtim): {n_exc}")

    parent = list(range(N))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in zip(*np.nonzero(easy)):
        parent[find(i)] = find(j)
    groups = {}
    for i in range(N):
        groups.setdefault(find(i), []).append(i)
    sizes = sorted((len(g) for g in groups.values()), reverse=True)
    print(f"\nKomponente të grafit të lehtë (kufiri teorik, rasti më optimist): {len(sizes)}")
    print("  madhësitë:", sizes[:25], "..." if len(sizes) > 25 else "")
    need = len(sizes) - 1
    print(f"  përjashtime të nevojshme të paktën: {need} | të lejuara: {E} -> "
          f"{'NEVOJA PLOTËSOHET' if need <= E else 'NUK PLOTËSOHET (nuk ka zgjidhje me këto rregulla)'}")

    t_end = min(t_max, 160.0)
    cpu = n_easy * (t_end / 0.02) * 0.047
    print(f"\nVlerësim për tabelën e kalimeve të lehta (hapi 0.02, deri në ditën {t_end:.0f}, 7 procese):")
    print(f"  ~ {cpu / 7 / 3600:.1f} orë | me hap 0.05: ~ {cpu / 7 / 3600 / 6:.1f} orë")
    print("  (bazuar në 'easy': 138 çifte, 7500 nisje -> rreth 1.9 orë)")


if __name__ == "__main__":
    main()