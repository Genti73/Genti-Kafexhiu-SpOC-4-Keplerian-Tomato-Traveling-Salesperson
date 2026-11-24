with open("lazy_beam.py", "r", encoding="utf-8") as f:
    src = f.read()


def replace_once(s, old, new, label):
    if s.count(old) != 1:
        print(f"STOP: blloku '{label}' nuk u gjet saktësisht një herë. Asgjë nuk u ndryshua.")
        raise SystemExit(1)
    return s.replace(old, new)


src = replace_once(src, "import json\nimport os\nimport sys\n", "import json\nimport os\nimport pickle\nimport sys\n", "import pickle")
src = replace_once(
    src,
    "ALPHA = float(sys.argv[5]) if len(sys.argv) > 5 else 3.0\n",
    "ALPHA = float(sys.argv[5]) if len(sys.argv) > 5 else 3.0\nRESUME = sys.argv[6] if len(sys.argv) > 6 else None\n",
    "parametri RESUME")
src = replace_once(src, "        ds, ts, tof_max, wait_max = 0.05, 0.02, 3.0, 3.0\n",
                   "        ds, ts, tof_max, wait_max = 0.05, 0.02, 3.0, 6.0\n", "pritja e përjashtimeve")
src = replace_once(
    src,
    """    states = {}
    for s in range(N):
        states[(s, 1 << s, 0)] = (t0, (s,), (), (), (), sigbit[s])
""",
    """    states = {}
    name = os.path.splitext(os.path.basename(INSTANCE))[0]
    if RESUME:
        with open(RESUME, "rb") as fh:
            states = pickle.load(fh)
        level_start = len(next(iter(states.values()))[1])
        print(f"Rinisje nga {RESUME}: {len(states)} gjendje me {level_start} domate të vizituara")
    else:
        for s in range(N):
            states[(s, 1 << s, 0)] = (t0, (s,), (), (), (), sigbit[s])
        level_start = 1
""",
    "gjendjet fillestare")
src = replace_once(src, "        for level in range(1, N):\n", "        for level in range(level_start, N):\n", "cikli i niveleve")
src = replace_once(
    src,
    "                kmin = min(prev, key=lambda k: prev[k][0])\n",
    """                with open(f"lazy_ckpt_{name}_death.pkl", "wb") as fh:
                    pickle.dump(prev, fh)
                print(f"  Gjendjet para vdekjes u ruajtën në lazy_ckpt_{name}_death.pkl")
                kmin = min(prev, key=lambda k: prev[k][0])
""",
    "ruajtja në vdekje")
src = replace_once(
    src,
    "            if level % 10 == 0 or level == N - 1:\n                ts = [v[0] for v in states.values()]\n",
    """            if level % 10 == 0:
                with open(f"lazy_ckpt_{name}_{level + 1}.pkl", "wb") as fh:
                    pickle.dump(states, fh)
            if level % 10 == 0 or level == N - 1:
                ts = [v[0] for v in states.values()]
""",
    "checkpoint")
src = replace_once(src, "        name = os.path.splitext(os.path.basename(INSTANCE))[0]\n        out = f\"best_lazy_{name}_f{fv:.3f}.json\"",
                   "        out = f\"best_lazy_{name}_f{fv:.3f}.json\"", "emri i dyfishtë")
with open("lazy_beam.py", "w", encoding="utf-8") as f:
    f.write(src)
print("OK: lazy_beam.py u përditësua (checkpoint, rinisje, pritje 6 ditë për përjashtime).")