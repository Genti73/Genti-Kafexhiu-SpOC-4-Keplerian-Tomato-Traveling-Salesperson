import json
from submission_helper import create_submission

x = json.load(open("solutions/best_polished_f116.740.json"))["x"]
create_submission("spoc-4-keplerian-tomato-traveling-salesperson", "small", x,
                  "submission_file.json", "Genti Kafexhiu", "DFS with memoization on precomputed edge tables + fine time refinement")