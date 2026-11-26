# Keplerian Tomato Traveling Salesperson (SpOC 4, Challenge 2)

Solution for the **Keplerian Tomato Traveling Salesperson Problem (KTTSP)** from ESA's
Space Optimisation Challenge 4 (SpOC 4, GECCO 2026).

- **Author:** Genti Kafexhiu (individual work)
- **Course:** Algorithm Design and Analysis (*Dizajni dhe Analiza e Algoritmeve*)
- **Problem statement and official files:** <https://github.com/esa/SpOC4> (Challenge 2)
- **Detailed write-up of the approach (Albanian):** [`docs/APPROACH_sq.md`](docs/APPROACH_sq.md)

## The problem in one paragraph

N tomatoes orbit the Moon on fixed Keplerian orbits. A spacecraft must visit every tomato exactly
once and finish as early as possible. A transfer from tomato *i* to tomato *j*, departing at time *t*
with time of flight *tof*, is a Lambert problem whose total ΔV depends on both *t* and *tof*. Every
transfer must cost at most ΔV_max = 100 m/s, except for at most E = 5 transfers that may cost up to
600 m/s. Waiting at a tomato is free. The objective is the arrival time at the last tomato.
It resembles a Traveling Salesperson Problem, but the edge costs are time-dependent and most edges
are infeasible.

## Results

| Instance | Tomatoes | Best total time f | Method | Status |
|---|---|---|---|---|
| easy (= *small*) | 49 | **116.74 days** | DFS with memoisation on precomputed edge tables, then continuous time refinement | valid (own validator) |
| easy, no tables | 49 | 123.03 days (118.13 with an earlier revision of the script) | lazy beam search (no precomputation) | valid (own validator) |
| medium | 181 | **221.92 days** | lazy beam search with plane-sweep rules | valid (own validator) |
| hard (= *large*) | 1051 | not solved | analysis only (see write-up) | - |

Solutions with f = 117.04 and f = 117.38 on *easy* were also confirmed with the official evaluator
class (`kttsp_udp.py`, copied from the ESA repository). All final solutions can be re-checked with the
commands in "Verification" below. Solutions are stored in [`solutions/`](solutions/).

For context, the public leaderboard for *small* shows the best scores around 101 days; this
repository's result was not ranked there.

## Main ideas

1. **A lower bound on ΔV from angular momentum.** A burn at radius *r* changes the angular momentum
   by at most *r*·|Δv|, and the Lambert arc conserves it. Hence
   ΔV ≥ |h_j − h_i| / r_max. This depends only on the orbital planes, not on time, so it cheaply
   discards almost all pairs (only ~6% of ordered pairs of *easy* can ever be cheap).
2. **Structure of the instances.** All tomatoes share (almost) the same node line, so the plane is
   set by the inclination. Tomatoes form three clusters (inclination ≈ 0, π/2, π), each with a group
   of low-orbit tomatoes, linked by a "staircase" of single tomatoes with equal inclination steps
   (6° in *easy*, 3° in *medium*). Low-orbit groups can only be entered and left with the special
   high-ΔV transfers, which explains why the budget of 5 exceptions is almost fully used.
3. **Earliest arrival is always best** (waiting is free), which gives a dominance rule that makes an
   exhaustive memoised search possible on precomputed tables.
4. **Continuous refinement.** For a fixed visiting order, taking the earliest possible arrival at
   every hop is optimal; a fine local search of departure time and time of flight recovers what a
   coarse table loses.
5. **For larger instances, evaluate transfers lazily** (only for a few candidate successors chosen
   with the lower bound) inside a beam search, with rules that exploit the staircase.

## Repository layout

The scripts import each other and read files with relative paths, so the layout is intentionally flat.

| Purpose | Files |
|---|---|
| Problem data | `problems/easy.kttsp`, `problems/medium.kttsp`, `problems/hard.kttsp` |
| Utilities (from ESA, header patched) | `utils_users.py`, `kttsp_udp.py` (official evaluator), `submission_helper.py` |
| Validation | `validator.py`, `check_official.py`, `check_margin.py` |
| Analysis of the problem | `explore1.py` ... `explore5.py`, `analyze_instance.py`, `diag_bridges.py`, `probe_hop.py`, `compare_hops.py`, `check_path_fine.py`, `show_windows.py`, `check_missing_cross.py` |
| Edge-table construction | `build_table.py`, `build_exc_table.py`, `build_exc_hh.py`, `build_fine.py`, `extend_fine.py` |
| Solvers | `static_search.py`, `solver.py`, `solver_beam.py`, `solver_dfs.py`, `harvest_polish.py`, `lazy_beam.py` |
| Refinement | `polish.py` |
| Submission | `make_submission.py` |
| One-off patch scripts | `patch_*.py` (kept as a record of the development steps) |
| Solutions | `solutions/*.json` |

## Reproducing the results

Environment (Windows, Anaconda/Miniconda):

```
conda create -n tomato python=3.10
conda activate tomato
conda install -c conda-forge "pykep<3" numpy matplotlib
```

`pykep` must be version 2.x.

**Quickest route (no precomputation), roughly 25 min for *easy* and 90 min for *medium*:**

```
python lazy_beam.py problems\easy.kttsp   10 6 4 3 - 0.04 1
python lazy_beam.py problems\medium.kttsp 10 6 6 3 - 0.04 1
```

Arguments after the instance: beam width per family, easy successors, structural exception
candidates, stranding penalty, resume file (`-` for none), interval tolerance, sweep on/off.
The script validates its result and writes `best_lazy_<instance>_f<value>.json`.

**Table-based route for *easy* (several hours of computation):**

```
python explore1.py                  # coarse minimum-ΔV matrix
python explore5.py                  # confirmed cheap pairs (easy_union.npz)
python build_table.py               # cheap-transfer table
python build_exc_table.py           # exception table, low <-> high orbit
python build_exc_hh.py              # exception table inside the high orbit
python build_fine.py easy           # finer tables (0.02 d)
python build_fine.py exc
python extend_fine.py easy          # horizon 150 d, flights up to 10 d
python extend_fine.py exc
# copy edge_table_*_fine2.npz to edge_table_easy.npz / edge_table_exc.npz
python solver_dfs.py 300            # exhaustive memoised search
python polish.py best_solution.json 0.01
```

## Verification

```
python validator.py solution.json problems\easy.kttsp          # own validator (prints every hop)
python check_official.py solution.json problems\easy.kttsp     # official evaluator class
```

Both check the permutation, the departure ordering, the ΔV limits with at most E exceptions and the
total time. Solutions keep a safety margin of 0.02 m/s below the ΔV limits.

## Limitations

- Results are heuristic. The memoised search is complete only when it exhausts the search space (the
  final runs were time-limited), and even then only with respect to the discretised edge tables, not
  the continuous problem. The lower-bound filter is a pruning device.
- The plane-sweep rules rely on the structure of the SpOC 4 instances (inclination staircase).
- *hard* (1051 tomatoes) was analysed but not solved: tabulating all candidate transfers would take
  an estimated 2000+ hours with the table method.

## Credits

The problem, instance files, `utils_users.py`, the evaluator class and the submission helper are by
ESA's Advanced Concepts Team (SpOC 4). All solver code, analysis and documentation in this repository
were written by the author.
