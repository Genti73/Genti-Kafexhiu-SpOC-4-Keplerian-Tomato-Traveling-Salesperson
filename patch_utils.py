with open("utils_users.py", "r", encoding="utf-8") as f:
    src = f.read()

replacements = [
    (
        'if len(parts) != 6 or parts[1].lower() != "kttsp":',
        'if len(parts) != 8 or parts[1].lower() != "kttsp":',
    ),
    (
        "        self.min_tof = float(header[3]) / pk.DAY2SEC\n"
        "        self.max_time = float(header[4])\n"
        "        self.dv_threshold = float(header[5])\n",
        "        self.min_tof = float(header[3])  # days\n"
        "        self.max_time = float(header[4])\n"
        "        self.dv_threshold = float(header[5])  # dV_max [m/s]\n"
        "        self.dv_exception = float(header[6])  # exception dV [m/s]\n"
        "        self.n_exceptions = int(float(header[7]))  # E\n",
    ),
]

for old, new in replacements:
    count = src.count(old)
    if count != 1:
        print("STOP: nuk u gjet saktësisht një herë ky tekst:")
        print(old)
        print("Gjetur:", count, "herë. Asgjë nuk u ndryshua.")
        raise SystemExit(1)
    src = src.replace(old, new)

with open("utils_users.py", "w", encoding="utf-8") as f:
    f.write(src)

print("OK: utils_users.py u përditësua.")