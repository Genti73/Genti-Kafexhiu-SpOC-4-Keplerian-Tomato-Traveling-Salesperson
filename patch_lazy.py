with open("lazy_beam.py", "r", encoding="utf-8") as f:
    src = f.read()

old = """                if used < E:
                    n_ex = 0
                    for j in exc_order[c]:
                        if (U >> j) & 1:
                            cl.append((j, 1))
                            n_ex += 1
                            if n_ex >= M_EXC:
                                break
"""
new = """                if used < E:
                    n_v = 0
                    for j in easy_order[c]:          # variante përjashtimi të kalimeve të lehta
                        if (U >> j) & 1:
                            cl.append((j, 1))
                            n_v += 1
                            if n_v >= K_EASY:
                                break
                    n_ex = 0
                    for j in exc_order[c]:           # kalime strukturore (kufiri teorik mbi dV_max)
                        if (U >> j) & 1 and LB[c, j] > dv_max:
                            cl.append((j, 1))
                            n_ex += 1
                            if n_ex >= M_EXC:
                                break
"""
if src.count(old) != 1:
    print("STOP: blloku nuk u gjet saktësisht një herë. Asgjë nuk u ndryshua.")
    raise SystemExit(1)
with open("lazy_beam.py", "w", encoding="utf-8") as f:
    f.write(src.replace(old, new))
print("OK: lazy_beam.py u përditësua.")