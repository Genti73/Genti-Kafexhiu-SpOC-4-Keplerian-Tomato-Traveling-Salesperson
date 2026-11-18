import time
from utils_users import TomatoProblem

util = TomatoProblem("problems/easy.kttsp")

for t in [0.0, 0.5, 1.0, 2.0]:
    for tof in [0.01, 0.1, 0.5, 1.0]:
        start = time.time()
        dv = util.compute_transfer(0, 1, t, tof)
        print(f"t={t}, tof={tof} -> dV = {dv}  ({time.time() - start:.4f} s)")