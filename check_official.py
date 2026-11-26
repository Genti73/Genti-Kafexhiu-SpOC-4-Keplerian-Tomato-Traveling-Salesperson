import json
import sys
from kttsp_udp import KTTSP

path = sys.argv[1] if len(sys.argv) > 1 else "best_solution.json"
instance = sys.argv[2] if len(sys.argv) > 2 else "problems/easy.kttsp"
udp = KTTSP(instance)
x = json.load(open(path))["x"]
f, perm, dv_c, time_c, exc_c = udp.fitness(x)
print(f"instanca = {instance} | f = {f:.4f}")
print(f"permutacioni = {perm} (duhet 0) | kufiri dV = {dv_c} (duhet 0) | "
      f"koha = {time_c} (duhet 0) | përjashtimet = {exc_c} (duhet <= 0)")
ok = perm == 0 and dv_c == 0 and time_c == 0 and exc_c <= 0
print("VLEFSHME SIPAS KODIT ZYRTAR" if ok else "NUK ËSHTË E VLEFSHME SIPAS KODIT ZYRTAR")