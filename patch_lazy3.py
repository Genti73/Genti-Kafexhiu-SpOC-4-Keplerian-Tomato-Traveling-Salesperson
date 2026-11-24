with open("lazy_beam.py", "r", encoding="utf-8") as f:
    src = f.read()

old = """                if dead > 1:
                    return False
        return True

    n_proc = max(1, cpu_count() - 1)
"""
new = """                if dead > 1:
                    return False
        # kontroll për komponente: çdo komponent i ndarë duhet të ketë hyrje dhe (përveç njërit) dalje
        cb = 1 << c
        rem = avail
        n_noexit = 0
        while rem:
            lb_ = rem & -rem
            comp, fr = lb_, lb_
            while fr:
                b = fr & -fr
                fr ^= b
                new = ue[b.bit_length() - 1] & rem & ~comp
                comp |= new
                fr |= new
            rem &= ~comp
            if comp & cb:
                continue
            outside = avail & ~comp
            can_in = can_out = False
            for u in bits(comp):
                if not can_in and (pe[u] | (px[u] if left > 0 else 0)) & outside:
                    can_in = True
                if not can_out and (se[u] | (sx[u] if left > 0 else 0)) & outside:
                    can_out = True
                if can_in and can_out:
                    break
            if not can_in:
                return False
            if not can_out:
                n_noexit += 1
                if n_noexit > 1:
                    return False
        return True

    n_proc = max(1, cpu_count() - 1)
"""
if src.count(old) != 1:
    print("STOP: blloku nuk u gjet saktësisht një herë. Asgjë nuk u ndryshua.")
    raise SystemExit(1)
with open("lazy_beam.py", "w", encoding="utf-8") as f:
    f.write(src.replace(old, new))
print("OK: lazy_beam.py u përditësua (kontroll hyrje/dalje për komponente).")