import sys
import numpy as np

a, b = int(sys.argv[1]), int(sys.argv[2])
for name in ("edge_table_easy.npz", "edge_table_exc.npz"):
    d = np.load(name)
    T, TOF = d["T"], d["TOF"]
    hit = [r for (i, j), r in zip(d["pairs"], d["tof_idx"]) if i == a and j == b]
    if not hit:
        print(f"{name}: çifti ({a},{b}) nuk është në tabelë")
        continue
    idx = hit[0]
    ok = idx >= 0
    print(f"{name}: disponueshmëria = {ok.mean() * 100:.1f}% ({int(ok.sum())} nga {len(ok)} nisje)")
    ks = np.nonzero(ok)[0]
    if len(ks) == 0:
        continue
    groups, s, p = [], ks[0], ks[0]
    for k in ks[1:]:
        if T[k] - T[p] > 0.3:
            groups.append((s, p))
            s = k
        p = k
    groups.append((s, p))
    print(f"  {len(groups)} intervale nisjesh (ditë) me fluturimin minimal në to:")
    for s, p in groups[:12]:
        print(f"    {T[s]:.2f} - {T[p]:.2f}  (fluturimi minimal {TOF[idx[s:p + 1]].min():.2f})")
    if len(groups) > 12:
        print(f"    ... edhe {len(groups) - 12} intervale, i fundit deri në {T[groups[-1][1]]:.2f}")