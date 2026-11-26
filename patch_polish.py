with open("polish.py", "r", encoding="utf-8") as f:
    src = f.read()
old = 'INSTANCE = "problems/easy.kttsp"\n'
new = 'INSTANCE = sys.argv[3] if len(sys.argv) > 3 else "problems/easy.kttsp"\n'
if src.count(old) != 1:
    print("STOP: rreshti nuk u gjet saktësisht një herë. Asgjë nuk u ndryshua.")
    raise SystemExit(1)
with open("polish.py", "w", encoding="utf-8") as f:
    f.write(src.replace(old, new))
print("OK: polish.py pranon tani instancën si argument të tretë.")
