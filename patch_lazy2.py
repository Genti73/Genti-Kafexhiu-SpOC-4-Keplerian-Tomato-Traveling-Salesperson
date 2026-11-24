with open("lazy_beam.py", "r", encoding="utf-8") as f:
    src = f.read()


def replace_once(s, old, new, label):
    if s.count(old) != 1:
        print(f"STOP: blloku '{label}' nuk u gjet saktësisht një herë. Asgjë nuk u ndryshua.")
        raise SystemExit(1)
    return s.replace(old, new)


src = replace_once(
    src,
    'M_EXC = int(float(sys.argv[4])) if len(sys.argv) > 4 else 2\n',
    'M_EXC = int(float(sys.argv[4])) if len(sys.argv) > 4 else 2\nALPHA = float(sys.argv[5]) if len(sys.argv) > 5 else 3.0\n',
    "parametri ALPHA")

src = replace_once(
    src,
    """            groups = {}
            for key2, v in children.items():
                groups.setdefault((v[5], key2[2]), []).append((v[0], key2, v))
            states = {}
""",
    """            groups = {}
            for key2, v in children.items():
                j2, vis2_, used2_ = key2
                U2 = full & ~vis2_
                near = U2 | (1 << j2)
                danger = 0
                for u in bits(U2):
                    if (ue[u] & near).bit_count() <= 1:
                        danger += 1
                groups.setdefault((v[5], key2[2]), []).append((v[0] + ALPHA * danger, key2, v))
            prev = states
            states = {}
""",
    "dënimi i ngecjes")

src = replace_once(
    src,
    """                print(f"  Niveli {level + 1}: asnjë gjendje nuk mbijetoi. Beam-i vdiq këtu.")
                return
""",
    """                print(f"  Niveli {level + 1}: asnjë gjendje nuk mbijetoi. Beam-i vdiq këtu.")
                kmin = min(prev, key=lambda k: prev[k][0])
                unv = [u for u in range(N) if not (kmin[1] >> u) & 1]
                print(f"  Gjendja më e mirë para vdekjes: {N - len(unv)} domate të vizituara, koha {prev[kmin][0]:.1f}, "
                      f"përjashtime {kmin[2]}")
                print(f"  Të pavizituara ({len(unv)}):", unv[:80])
                return
""",
    "diagnostikimi")

with open("lazy_beam.py", "w", encoding="utf-8") as f:
    f.write(src)
print("OK: lazy_beam.py u përditësua (dënimi i ngecjes + diagnostikim).")