# d_invariant — Brieskorn sphere d-invariants

Computes the Heegaard Floer d-invariant of the Brieskorn sphere Σ(p, q, r),
where p, q, r are pairwise coprime positive integers.

## Formula

$$d(\Sigma(p,q,r)) = \frac{K^2+s}{4} - 2\min_{0 \le m \le pqr} \tau(m)$$

- **K²+s** is assembled from the Seifert invariants via Borodzik–Nemethi (2014), eq. (2.4),
  using three classical Dedekind sums.
- **τ(m)** is the correction term: τ(0) = 0, τ(m) = τ(m−1) + δ_{m−1}, where
  δ_j = 1 − j·e₀ − ⌈j·ω₀/p⌉ − ⌈j·ω₁/q⌉ − ⌈j·ω₂/r⌉.
- The Seifert invariants (e₀, ω₀, ω₁, ω₂) are determined by the modular inverses
  of the pairwise products of p, q, r.

## Build

Requires: FLINT 3, GMP, a C++17 compiler, OpenMP (optional but recommended).

```bash
g++ -O3 -march=native -fopenmp \
    -I/home/josefsvoboda/local/flint3/include \
    -L/home/josefsvoboda/local/flint3/lib \
    -Wl,-rpath,/home/josefsvoboda/local/flint3/lib \
    d_invariant.cpp -lflint -lgmp -o d_invariant
```

## Usage

**Single triple:**
```
./d_invariant p q r
```
Prints d, K²+s/4, and 2·min τ.

**Batch scan — all pairwise-coprime triples p < q < r < N:**
```
./d_invariant --scan N
./d_invariant --scan N -j 8     # 8 OpenMP threads
```
Outputs CSV to stdout: `p,q,r,d_invariant,K^2+s/4,2*min_tau`

**Examples:**
```
$ ./d_invariant 2 3 13
Σ(2,3,13):
  d          = 0
  K²+s/4     = 0
  2·min_tau  = 0

$ ./d_invariant --scan 50 -j 14 > results.csv
```

## Algorithm

### Seifert invariants
For α = [p, q, r] and α̂ = [qr, pr, pq]:

    ωᵢ = (−α̂ᵢ⁻¹) mod αᵢ,    0 < ωᵢ < αᵢ
    e₀ = (−1 − Σ ωᵢ α̂ᵢ) / (pqr)     # always −1, −2, or −3

Modular inverses are computed with `fmpz_invmod` (FLINT).

### K²+s (exact rational arithmetic)
    e       = e₀ + Σ(ωᵢ/αᵢ)
    ε       = (−1 + Σ(1/αᵢ)) / e
    K²+s    = ε²·e + e + 5 − 12·Σ s(ωᵢ, αᵢ)

The three Dedekind sums s(ωᵢ, αᵢ) are computed exactly as rationals with
`fmpq_dedekind_sum` (FLINT). The entire expression is assembled in exact rational
arithmetic and verified to be an integer divisible by 4 before extracting K²+s/4.
This avoids the floating-point rounding used in the reference Python implementation.

### τ minimum — division-free streaming loop
The dominant cost is O(p·q·r) per triple. Two key improvements over the
Python/NumPy reference:

**1. O(1) memory.** Python allocates a full array of size p·q·r+1.
The C++ version maintains only two scalars (running τ and running minimum).

**2. No integer divisions in the hot loop.** The Python reference computes
⌈j·ω/α⌉ = (j·ω + α − 1) / α — one division per arm per step. Instead, we track:

    rem[i] = (j · ωᵢ) mod αᵢ
    flr[i] = (j · ωᵢ) div αᵢ

and then ⌈j·ωᵢ/αᵢ⌉ = flr[i] + (rem[i] > 0 ? 1 : 0).
Each step updates: `rem[i] += ω[i]; if (rem[i] >= α[i]) { rem[i] -= α[i]; flr[i]++; }`

The inner loop contains only additions and comparisons — no div instructions.

### Batch parallelism
Triple enumeration is sequential; d-invariant computation is parallelised with
OpenMP (`schedule(dynamic, 64)` over the flat triple list). Output is in
deterministic order regardless of thread count.

## Performance (Intel Core Ultra 5 125U, 14 logical cores)

| Scan range | Triples | 1 thread | 14 threads |
|------------|---------|----------|------------|
| N < 45     | 3,668   | 0.15 s   | —          |
| N < 100    | 43,519  | 17.6 s   | 3.9 s      |
| N < 200    | 77,558  | ~19 min  | ~4 min     |

Scaling is O(N⁶): both the number of triples and the average loop length grow as N³.
The Python reference takes ~1 hour for N < 200.

## Correctness

All 3,668 values in `brieskorn_d_all_computed.csv` and all 77,558 values in
`brieskorn_d_all_200.csv` match exactly.
