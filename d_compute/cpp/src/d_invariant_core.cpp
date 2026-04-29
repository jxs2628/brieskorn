// d_invariant_core.cpp
//
// Implementations of the d-invariant computation functions for Brieskorn
// spheres Σ(p, q, r).  See d_invariant_core.h for the public API, and
// d_invariant.cpp for the full mathematical description and build instructions.

#include "d_invariant_core.h"

#include <algorithm>   // std::swap (used in gcd64)
#include <iostream>
#include <cstdlib>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef __AVX2__
#include <immintrin.h>
#endif

// FLINT: exact-arithmetic library.  The 'extern "C"' block is needed because
// FLINT is a C library, not C++.
extern "C" {
#include <flint/fmpz.h>   // fmpz_t  — exact integer type
#include <flint/fmpq.h>   // fmpq_t  — exact rational type
}

// ─────────────────────────────────────────────────────────────────────────────
// Greatest common divisor (Euclidean algorithm)
// ─────────────────────────────────────────────────────────────────────────────
i64 gcd64(i64 a, i64 b) {
    while (b) { a %= b; std::swap(a, b); }
    return a;
}

// ─────────────────────────────────────────────────────────────────────────────
// Seifert invariants for Σ(p, q, r)
//
// Every Seifert-fibered rational homology sphere is encoded by:
//   • e0       — the "orbifold Euler number" integer part
//   • alpha[i] — the orders of the three exceptional fibers (= p, q, r)
//   • omega[i] — the "twisting" at the i-th exceptional fiber; an integer
//                satisfying  0 < omega[i] < alpha[i]  with the normalization
//                omega[i] ≡ −(alpha_hat[i])^{−1}  (mod alpha[i]),
//                where  alpha_hat[i] = product of the other two alphas.
//
// The total Euler number is  e = e0 + Σ omega[i]/alpha[i].
// ─────────────────────────────────────────────────────────────────────────────
SeifertData brieskorn_seifert(i64 p, i64 q, i64 r) {
    SeifertData sd;
    sd.alpha[0] = p;
    sd.alpha[1] = q;
    sd.alpha[2] = r;

    // alpha_hat[i] is the product of the two alphas other than alpha[i].
    // For Σ(p,q,r): alpha_hat = {q*r, p*r, p*q}.
    const i64 alpha_hat[3] = { q*r, p*r, p*q };

    // FLINT integers used for modular arithmetic.
    // (fmpz_t is FLINT's exact integer type; must be initialised before use
    //  and cleared afterwards to free memory.)
    fmpz_t fa, fb, finv;
    fmpz_init(fa); fmpz_init(fb); fmpz_init(finv);

    for (int i = 0; i < 3; i++) {
        fmpz_set_si(fa, alpha_hat[i]);
        fmpz_set_si(fb, sd.alpha[i]);

        // Compute  finv = alpha_hat[i]^{−1}  (mod alpha[i]).
        // This is the unique integer x in {0,...,alpha[i]−1} such that
        // alpha_hat[i] * x ≡ 1 (mod alpha[i]).
        fmpz_invmod(finv, fa, fb);

        const i64 inv_val = fmpz_get_si(finv);

        // omega[i] = (−inv_val) mod alpha[i], shifted into the range (0, alpha[i]).
        // The shift ensures 0 < omega[i] < alpha[i] (strictly positive).
        sd.omega[i] = sd.alpha[i] - inv_val;
    }

    fmpz_clear(fa); fmpz_clear(fb); fmpz_clear(finv);

    // Compute e0 from the relation:
    //   e0 * (p*q*r) = −1 − Σ omega[i] * alpha_hat[i]
    // This must be an exact integer (guaranteed by the theory).
    i64 rhs = -1;
    for (int i = 0; i < 3; i++) rhs -= sd.omega[i] * alpha_hat[i];
    sd.e0 = rhs / (p * q * r);

    return sd;
}

// ─────────────────────────────────────────────────────────────────────────────
// Compute  (K² + s) / 4  via exact rational arithmetic
//
// K² + s  is computed from the Seifert data using the formula
// (Borodzik–Nemethi 2014, eq. 2.4):
//
//   K² + s = ε² · e + e + 5 − 12 · Σ s(ω_i, α_i)
//
// where:
//   e   = e0 + Σ(ω_i / α_i)          (total Euler number of the Seifert fibration)
//   ε   = (−1 + Σ(1/α_i)) / e         (a correction factor; ν=3 exceptional fibers
//                                       so the formula uses (2−ν + ...) = (−1 + ...))
//   s(h,k) is the classical Dedekind sum
//
// The result is guaranteed to be an integer, so we divide by 4 and return
// an i64.  Using exact rational arithmetic (FLINT fmpq_t) avoids any
// floating-point rounding error.
// ─────────────────────────────────────────────────────────────────────────────
i64 compute_ks_term(const SeifertData& sd) {
    // fmpq_t is FLINT's exact rational number type (numerator/denominator pair).
    fmpq_t seifert_euler_number, epsilon_numerator, epsilon;
    fmpq_t ks, dedekind_sum_total, tmp;

    fmpq_init(seifert_euler_number);
    fmpq_init(epsilon_numerator);
    fmpq_init(epsilon);
    fmpq_init(ks);
    fmpq_init(dedekind_sum_total);
    fmpq_init(tmp);

    // ── Step 1: seifert_euler_number = e0 + Σ(omega[i] / alpha[i]) ──────────
    fmpq_set_si(seifert_euler_number, (slong)sd.e0, 1u);
    for (int i = 0; i < 3; i++) {
        // Add the fraction omega[i] / alpha[i] to seifert_euler_number.
        fmpq_set_si(tmp, (slong)sd.omega[i], (ulong)sd.alpha[i]);
        fmpq_add(seifert_euler_number, seifert_euler_number, tmp);
    }

    // ── Step 2: epsilon = (−1 + Σ(1/alpha[i])) / seifert_euler_number ───────
    // For a Seifert fibration with ν=3 exceptional fibers the numerator is
    // (2 − ν + Σ 1/α_i) = (−1 + Σ 1/α_i).
    fmpq_set_si(epsilon_numerator, -1, 1u);
    for (int i = 0; i < 3; i++) {
        fmpq_set_si(tmp, 1, (ulong)sd.alpha[i]);
        fmpq_add(epsilon_numerator, epsilon_numerator, tmp);
    }
    fmpq_div(epsilon, epsilon_numerator, seifert_euler_number);

    // ── Step 3: ks = ε² · e + e + 5 ─────────────────────────────────────────
    fmpq_mul(ks, epsilon, epsilon);             // ks = ε²
    fmpq_mul(ks, ks, seifert_euler_number);     // ks = ε² · e
    fmpq_add(ks, ks, seifert_euler_number);     // ks = ε² · e + e
    fmpq_set_si(tmp, 5, 1u);
    fmpq_add(ks, ks, tmp);                      // ks = ε² · e + e + 5

    // ── Step 4: subtract  12 · Σ s(omega[i], alpha[i])  ─────────────────────
    // s(h, k) is the Dedekind sum; FLINT computes it exactly as a rational.
    fmpq_zero(dedekind_sum_total);
    {
        fmpz_t h, k;
        fmpz_init(h); fmpz_init(k);
        for (int i = 0; i < 3; i++) {
            fmpz_set_si(h, sd.omega[i]);
            fmpz_set_si(k, sd.alpha[i]);
            fmpq_dedekind_sum(tmp, h, k);
            fmpq_add(dedekind_sum_total, dedekind_sum_total, tmp);
        }
        fmpz_clear(h); fmpz_clear(k);
    }

    fmpq_set_si(tmp, 12, 1u);
    fmpq_mul(tmp, tmp, dedekind_sum_total);     // tmp = 12 · Σ s(ω,α)
    fmpq_sub(ks, ks, tmp);                      // ks = K² + s

    // ── Step 5: divide by 4 ──────────────────────────────────────────────────
    fmpq_set_si(tmp, 1, 4u);
    fmpq_mul(ks, ks, tmp);                      // ks = (K² + s) / 4

    // Sanity check: the theory guarantees this is an integer.
    // If the denominator is not 1, something has gone wrong in the computation.
    if (!fmpz_is_one(fmpq_denref(ks))) {
        std::cerr << "ERROR: (K²+s)/4 is not an integer for Σ("
                  << sd.alpha[0] << ',' << sd.alpha[1] << ',' << sd.alpha[2] << ")\n";
        std::exit(1);
    }

    const i64 result = fmpz_get_si(fmpq_numref(ks));

    fmpq_clear(seifert_euler_number);
    fmpq_clear(epsilon_numerator);
    fmpq_clear(epsilon);
    fmpq_clear(ks);
    fmpq_clear(dedekind_sum_total);
    fmpq_clear(tmp);

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Compute  min{ τ(m) : m = 0, 1, ..., N₀ }   where N₀ = pqr − pq − qr − rp.
//
// τ is the "tau function" appearing in the d-invariant formula.
// It is defined as a running sum:
//
//   τ(0) = 0
//   τ(m) = τ(m−1) + δ_{m−1}
//
// where each increment δ_j is:
//
//   δ_j = 1 − j·e0 − Σ_i ⌈ j·ω_i / α_i ⌉
//
// Intuitively, δ_j counts a "lattice gap" at step j; its exact combinatorial
// meaning comes from the Seifert surgery description.
//
// Scan range: by Karakurt (Thm 1.3), Δ(n) = τ(n+1) − τ(n) ≥ 0 for n > N₀,
// so τ is non-decreasing past N₀ and its global minimum lies in [0, N₀].
// (For Σ(2,3,5), N₀ = −1 < 0; the formula degenerates and the minimum is τ(0)=0.)
//
// ── Performance trick ────────────────────────────────────────────────────────
// Computing ⌈j·ω_i/α_i⌉ naively requires an integer division for every j,
// which is expensive.  Instead we maintain, for each i:
//
//   remainder[i]  = (j · omega[i]) mod alpha[i]     — the fractional part
//   floor_part[i] = (j · omega[i]) div alpha[i]     — the integer part
//
// Then  ⌈j·ω_i/α_i⌉ = floor_part[i] + (remainder[i] > 0 ? 1 : 0).
//
// When j increases by 1, the update is just:
//   remainder[i] += omega[i]
//   if (remainder[i] >= alpha[i]) { remainder[i] -= alpha[i]; floor_part[i]++; }
//
// This replaces every division with an addition and a comparison.
// ─────────────────────────────────────────────────────────────────────────────
i64 compute_min_tau(const SeifertData& sd) {
    // Scan range: m_max = max(0, N₀)  where  N₀ = pqr − pq − qr − pr.
    // Karakurt: τ is non-decreasing past N₀, so its global minimum lies in [0, N₀].
    // For Σ(2,3,5), N₀ = −1; we clamp to 0 so the τ(0)=0 record is still taken.
    const i64 N0 = sd.alpha[0] * sd.alpha[1] * sd.alpha[2]
                 - sd.alpha[0] * sd.alpha[1]
                 - sd.alpha[1] * sd.alpha[2]
                 - sd.alpha[2] * sd.alpha[0];
    const i64 m_max = N0 > 0 ? N0 : 0;

    // State for the division-free incremental computation.
    i64 remainder[3]  = {0, 0, 0};   // (j * omega[i]) mod alpha[i]
    i64 floor_part[3] = {0, 0, 0};   // (j * omega[i]) div alpha[i]
    i64 j_times_e0    = 0;           // j * e0, incremented by e0 each step
    i64 tau           = 0;           // τ(j), the running sum
    i64 min_tau       = 0;           // running minimum of τ

    for (i64 j = 0; j <= m_max; j++) {
        // Record τ(j) and update the running minimum.
        if (tau < min_tau) min_tau = tau;

        // We've recorded the last value; exit before computing a spurious δ.
        if (j == m_max) break;

        // Compute δ_j = 1 − j·e0 − Σ_i ⌈j·ω_i/α_i⌉
        // using the precomputed floor_part and remainder.
        i64 delta = 1 - j_times_e0;
        for (int i = 0; i < 3; i++) {
            // ⌈j·ω_i/α_i⌉ = floor_part[i] + (1 if there's a nonzero remainder)
            delta -= floor_part[i] + (remainder[i] > 0 ? 1LL : 0LL);
        }
        tau += delta;

        // Advance the incremental state from j to j+1.
        j_times_e0 += sd.e0;
        for (int i = 0; i < 3; i++) {
            remainder[i] += sd.omega[i];
            if (remainder[i] >= sd.alpha[i]) {
                remainder[i] -= sd.alpha[i];
                floor_part[i]++;
            }
        }
    }

    return min_tau;
}

// ─────────────────────────────────────────────────────────────────────────────
// compute_tau_sequence — return τ(0), τ(1), …, τ(N₀) as a vector.
//
// Same incremental loop as compute_min_tau; instead of tracking the running
// minimum we record every value.  Intended for studying τ on a single triple,
// not for batch scans (allocates N₀+1 entries).
// ─────────────────────────────────────────────────────────────────────────────
std::vector<i64> compute_tau_sequence(const SeifertData& sd) {
    // m_max = max(0, N₀)  where  N₀ = pqr − pq − qr − rp.
    // Karakurt: τ is non-decreasing past N₀, so its global minimum lies in [0, N₀].
    // For Σ(2,3,5), N₀ = −1; we clamp to 0 so the τ(0)=0 record is still taken.
    const i64 N0 = sd.alpha[0] * sd.alpha[1] * sd.alpha[2]
                 - sd.alpha[0] * sd.alpha[1]
                 - sd.alpha[1] * sd.alpha[2]
                 - sd.alpha[2] * sd.alpha[0];
    const i64 m_max = N0 > 0 ? N0 : 0;

    std::vector<i64> tau_seq;
    tau_seq.resize((size_t)m_max + 1);

    i64 remainder[3]  = {0, 0, 0};
    i64 floor_part[3] = {0, 0, 0};
    i64 j_times_e0    = 0;
    i64 tau           = 0;

    for (i64 j = 0; j <= m_max; j++) {
        tau_seq[(size_t)j] = tau;
        if (j == m_max) break;

        i64 delta = 1 - j_times_e0;
        for (int i = 0; i < 3; i++) {
            delta -= floor_part[i] + (remainder[i] > 0 ? 1LL : 0LL);
        }
        tau += delta;

        j_times_e0 += sd.e0;
        for (int i = 0; i < 3; i++) {
            remainder[i] += sd.omega[i];
            if (remainder[i] >= sd.alpha[i]) {
                remainder[i] -= sd.alpha[i];
                floor_part[i]++;
            }
        }
    }

    return tau_seq;
}

// ─────────────────────────────────────────────────────────────────────────────
// compute_min_tau_semigroup — Karakurt-style min τ via lattice enumeration.
//
// Theorem 1.3 (Can–Karakurt 2014) characterizes Δ(n) = τ(n+1) − τ(n) on
// [0, N₀] where N₀ = pqr − pq − qr − pr:
//
//   • Δ(n) ∈ {−1, 0, 1}.
//   • Δ(n) = +1  iff  n ∈ G(pq, pr, qr)  (numerical semigroup).
//   • Δ(n) = −Δ(N₀ − n)   ⇒   Δ(n) = −1  iff  n ∉ G  and  N₀ − n ∈ G.
//   • Δ(n) ≥ 0 for n > N₀.
//
// τ therefore changes only at positions in G ∩ [0, N₀] (up by 1) and
// (N₀ − G) ∩ [0, N₀] (down by 1).  These two sets are disjoint: if
// n ∈ G and N₀ − n ∈ G simultaneously, then Δ(n) = +1 (item 5) and
// Δ(n) = −Δ(N₀ − n) = −1, contradiction.
//
// Algorithm:
//   1. Enumerate G ∩ [0, N₀] by triple loop a·pq + b·pr + c·qr ≤ N₀.
//      Representations are not unique (e.g., r·pq = q·pr = p·qr = pqr),
//      so the list must be sorted and de-duplicated.
//   2. Merge two sorted streams: G[i] ascending (each +1) and the down
//      events {N₀ − G[k−1−i]} ascending (each −1).  Track running τ and
//      its minimum.
//
// Complexity: O(|G ∩ [0, N₀]|) work and memory, where |G ∩ [0, N₀]| is
// roughly N₀³ / (6·pq·pr·qr) — much smaller than the O(pqr) walk used by
// the incremental routines.
//
// Excluded case: (p,q,r) = (2,3,5) gives N₀ = −1; fall back to compute_min_tau.
// ─────────────────────────────────────────────────────────────────────────────
i64 compute_min_tau_semigroup(i64 p, i64 q, i64 r) {
    const i64 pq = p * q, pr = p * r, qr = q * r;
    const i64 N0 = p * q * r - pq - qr - pr;
    if (N0 <= 0) {
        // Only Σ(2,3,5); fall back to the general loop.
        return compute_min_tau(brieskorn_seifert(p, q, r));
    }

    // Enumerate G ∩ [0, N₀] over the lattice tetrahedron a·pq + b·pr + c·qr ≤ N₀.
    std::vector<i64> G;
    G.reserve((size_t)((double)N0 * N0 * N0 / (6.0 * pq * pr * qr)) + 16);
    for (i64 a = 0; a * pq <= N0; a++) {
        const i64 base_a = a * pq;
        for (i64 b = 0; base_a + b * pr <= N0; b++) {
            const i64 base_ab = base_a + b * pr;
            for (i64 c = 0; base_ab + c * qr <= N0; c++) {
                G.push_back(base_ab + c * qr);
            }
        }
    }
    std::sort(G.begin(), G.end());
    G.erase(std::unique(G.begin(), G.end()), G.end());

    // Merge sorted up-events (G ascending, +1) with down-events (N₀ − G in
    // ascending order, −1).  The smallest down position is N₀ − G.back();
    // the largest is N₀ − G.front() = N₀.  By disjointness next_up != next_down.
    const size_t k = G.size();
    const i64 sentinel = N0 + 1;
    i64 tau = 0, min_tau = 0;
    size_t up = 0;       // next +1 at G[up]
    size_t down = k;     // next −1 at N₀ − G[down − 1]
    while (up < k || down > 0) {
        const i64 next_up   = (up < k)   ? G[up]              : sentinel;
        const i64 next_down = (down > 0) ? (N0 - G[down - 1]) : sentinel;
        if (next_up < next_down) {
            tau += 1;
            up++;
        } else {
            tau -= 1;
            down--;
            if (tau < min_tau) min_tau = tau;
        }
    }
    return min_tau;
}

// ─────────────────────────────────────────────────────────────────────────────
// floor_sum and tau_at — O(log α) random access to τ
//
// floor_sum_impl(n, a, b, m) computes  Σⱼ₌₀ⁿ⁻¹ ⌊(a·j + b) / m⌋.
//
// Algorithm (lattice-point swap, structurally identical to Euclid's algorithm):
//
//   1. Extract the integer part of a/m: every term gets an extra ⌊a/m⌋·j,
//      contributing ⌊a/m⌋ · n(n−1)/2 to the total.  Replace a ← a mod m.
//
//   2. Extract the integer part of b/m: every term gets an extra ⌊b/m⌋,
//      contributing ⌊b/m⌋ · n to the total.  Replace b ← b mod m.
//
//   3. Now 0 ≤ a < m and 0 ≤ b < m.  Let y_max = ⌊(a·n + b)/m⌋.
//      Counting the same lattice points from the other axis:
//
//      Σⱼ₌₀ⁿ⁻¹ ⌊(a·j+b)/m⌋  =  y_max·(n−1)  −  Σₖ₌₀^{y_max−1} ⌊(m·k + m−b−1)/a⌋
//                             =  y_max·(n−1)  −  floor_sum_impl(y_max, m, m−b−1, a)
//
//      The recursion terminates because the pair (m, a) shrinks by the
//      Euclidean algorithm: after each swap, new denominator = old a < old m.
//
// Requires: n ≥ 0, m > 0, a ≥ 0, b ≥ 0.
// ─────────────────────────────────────────────────────────────────────────────
static i64 floor_sum_impl(i64 n, i64 a, i64 b, i64 m) {
    if (n == 0) return 0;
    i64 ans = 0;

    // Step 1: reduce a
    if (a >= m) {
        ans += (a / m) * (n * (n - 1) / 2);
        a   %= m;
    }
    // Step 2: reduce b
    if (b >= m) {
        ans += (b / m) * n;
        b   %= m;
    }
    // Now 0 ≤ a < m, 0 ≤ b < m.
    const i64 y_max = (a * n + b) / m;
    if (y_max == 0) return ans;

    // Lattice-point swap (the Euclidean step):
    ans += y_max * (n - 1) - floor_sum_impl(y_max, m, m - b - 1, a);
    return ans;
}

// Public wrapper: Σⱼ₌₀ⁿ⁻¹ ⌊j·h/k⌋
i64 floor_sum(i64 n, i64 h, i64 k) {
    return floor_sum_impl(n, h, 0, k);
}

// ─────────────────────────────────────────────────────────────────────────────
// tau_at(m, sd) — exact value of τ(m) for any m, in O(log α) time.
//
// Derivation: τ(m) = Σⱼ₌₀ᵐ⁻¹ δⱼ  where  δⱼ = 1 − j·e0 − Σᵢ ⌈j·ωᵢ/αᵢ⌉.
//
// Summing term by term:
//   • Σⱼ 1          =  m
//   • Σⱼ j·e0       =  e0 · m(m−1)/2
//   • Σⱼ ⌈j·ω/α⌉   =  floor_sum(m, ω, α)  +  (m−1)  −  ⌊(m−1)/α⌋
//     The extra (m−1) − ⌊(m−1)/α⌋ counts j ∈ {1,...,m−1} not divisible by α,
//     i.e. the j for which ⌈j·ω/α⌉ = ⌊j·ω/α⌋ + 1 rather than ⌊j·ω/α⌋.
//     (Valid because gcd(ω, α) = 1.)
// ─────────────────────────────────────────────────────────────────────────────
i64 tau_at(i64 m, const SeifertData& sd) {
    if (m == 0) return 0;
    i64 result = m;
    result -= sd.e0 * m * (m - 1) / 2;
    for (int i = 0; i < 3; i++) {
        result -= floor_sum(m, sd.omega[i], sd.alpha[i]);
        result -= (m - 1) - (m - 1) / sd.alpha[i];  // integer division = floor for m≥1, α≥2
    }
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// compute_min_tau_parallel — same result as compute_min_tau, but splits
// [0, N₀] into num_threads equal chunks processed in parallel.
//
// Each thread:
//   1. Computes τ(start) via tau_at — O(log α).
//   2. Reconstructs the incremental state at j = start in O(1):
//        remainder[i] = (start · ωᵢ) mod αᵢ
//        floor_part[i] = (start · ωᵢ) div αᵢ
//        j_times_e0    = start · e0
//   3. Runs the same inner loop as compute_min_tau over [start, end].
//   4. Reports its local minimum; the global minimum is the smallest of these.
// ─────────────────────────────────────────────────────────────────────────────
i64 compute_min_tau_parallel(const SeifertData& sd, int num_threads) {
    // m_max = max(0, N₀)  where  N₀ = pqr − pq − qr − rp.
    // Karakurt: τ is non-decreasing past N₀, so its global minimum lies in [0, N₀].
    // For Σ(2,3,5), N₀ = −1; we clamp to 0 so the τ(0)=0 record is still taken.
    const i64 N0 = sd.alpha[0] * sd.alpha[1] * sd.alpha[2]
                 - sd.alpha[0] * sd.alpha[1]
                 - sd.alpha[1] * sd.alpha[2]
                 - sd.alpha[2] * sd.alpha[0];
    const i64 m_max = N0 > 0 ? N0 : 0;

    i64 global_min = 0;

#ifdef _OPENMP
    omp_set_num_threads(num_threads);
#endif

    #pragma omp parallel reduction(min : global_min) num_threads(num_threads)
    {
        // Determine this thread's slice [start, end].
#ifdef _OPENMP
        const int t    = omp_get_thread_num();
        const int T    = omp_get_num_threads();
#else
        const int t    = 0;
        const int T    = 1;
#endif
        const i64 start = (i64)t       * m_max / T;
        const i64 end   = (i64)(t + 1) * m_max / T;

        // --- Step 1: jump to τ(start) via the closed-form formula ---
        i64 tau = tau_at(start, sd);

        // --- Step 2: reconstruct the incremental state at j = start ---
        i64 remainder[3], floor_part[3];
        for (int i = 0; i < 3; i++) {
            const i64 prod  = start * sd.omega[i];
            remainder[i]    = prod % sd.alpha[i];
            floor_part[i]   = prod / sd.alpha[i];
        }
        i64 j_times_e0 = start * sd.e0;

        // --- Step 3: run the same incremental loop as compute_min_tau ---
        i64 local_min = tau;

        for (i64 j = start; j <= end; j++) {
            if (tau < local_min) local_min = tau;
            if (j == end) break;

            i64 delta = 1 - j_times_e0;
            for (int i = 0; i < 3; i++)
                delta -= floor_part[i] + (remainder[i] > 0 ? 1LL : 0LL);
            tau += delta;

            j_times_e0 += sd.e0;
            for (int i = 0; i < 3; i++) {
                remainder[i] += sd.omega[i];
                if (remainder[i] >= sd.alpha[i]) {
                    remainder[i] -= sd.alpha[i];
                    floor_part[i]++;
                }
            }
        }

        global_min = local_min;  // reduction(min:) takes care of the rest
    }

    return global_min;
}

// ─────────────────────────────────────────────────────────────────────────────
// Assemble the final formula:  d = (K² + s)/4  −  2 · min τ
// ─────────────────────────────────────────────────────────────────────────────
DResult d_invariant(i64 p, i64 q, i64 r) {
    const SeifertData sd = brieskorn_seifert(p, q, r);
    const i64 ks_term    = compute_ks_term(sd);
    const i64 min_tau    = compute_min_tau(sd);
    return { ks_term - 2*min_tau, ks_term, 2*min_tau };
}

// ─────────────────────────────────────────────────────────────────────────────
// build_ceil_tables — populate CeilTableCache for all (omega, alpha) with
// 1 ≤ omega < alpha ≤ N.
//
// Storage layout: for each alpha from 2 to N, we reserve (alpha−1)*alpha
// entries in the flat array.  The block for a given alpha starts at
// offsets[alpha].  Within that block, the row for omega starts at
// (omega−1)*alpha, giving alpha entries:  entry[k] = ⌈k·omega/alpha⌉.
//
// ⌈k·omega/alpha⌉ = (k·omega + alpha − 1) / alpha  (exact integer division).
// ─────────────────────────────────────────────────────────────────────────────
CeilTableCache build_ceil_tables(int N) {
    CeilTableCache cache;
    cache.offsets.resize(N + 1, 0);

    // First pass: compute total data size and set offsets.
    size_t total = 0;
    for (int a = 2; a <= N; a++) {
        cache.offsets[a] = total;
        total += (size_t)(a - 1) * (size_t)a;
    }
    cache.data.resize(total);

    // Second pass: fill each table row.
    for (int a = 2; a <= N; a++) {
        for (int w = 1; w < a; w++) {
            int16_t* tbl = cache.data.data() + cache.offsets[a]
                         + (size_t)(w - 1) * (size_t)a;
            for (int k = 0; k < a; k++) {
                // ⌈k·w/a⌉  using exact integer arithmetic (no floating point)
                tbl[k] = static_cast<int16_t>((k * static_cast<int64_t>(w) + a - 1) / a);
            }
        }
    }
    return cache;
}

// ─────────────────────────────────────────────────────────────────────────────
// process_chunk — inner kernel called by compute_min_tau_fast.
//
// Processes L delta steps with NO phase wraps (guaranteed by the caller).
// At step k (relative), the delta is:
//
//   δ[k] = C − k·e0 − t0[k] − t1[k] − t2[k]
//
// where C = 1 − j_base·e0 − Σᵢ block_offset[i]  (constant for this chunk)
// and   tᵢ = cache pointer already advanced to the current phase for component i.
//
// tau is accumulated sequentially (each step depends on the previous), and
// min_tau is updated after every step.  The FUNCTION RECORDS tau[k+1] (after
// each delta); tau[j_base] was already recorded by the caller before this call.
//
// ── AVX2 path: 8 deltas per batch with int32 SIMD ────────────────────────────
//
// Two properties of this kernel let us use the wider 8-wide int32 path
// instead of a 4-wide int64 path:
//
// (1) int32_t is sufficient for every value used in the SIMD computation.
//
//     A short identity (using the integer-homology-sphere relation
//     e = e₀ + Σ ωᵢ/αᵢ = 1/pqr) gives
//
//         C = 1 − j·e₀ − Σ block_offset[i]
//           = 1 − j/pqr + Σ phase_i·ωᵢ/αᵢ,
//
//     so 0 ≤ C ≤ 1 + 3·(N−1) ≈ 3N.  At N=700 this is ≤ ~2100.
//     Each lane is C − lane·e₀ with |e₀| ≤ 3, and after each batch every
//     lane shifts by 8·e₀, but k ≤ αᵢ − 1 ≤ N, so |k·e₀| ≤ 3N.  Table
//     entries tᵢ[k] ≤ αᵢ − 1 ≤ N − 1.  Every operand fits comfortably in
//     int32 (|·| ≪ 2^31) for any realistic N.
//
//     This lets us load 8 int16_t values per component in one __m128i and
//     widen to a single __m256i of int32, so each batch advances 8 steps
//     (vs. 4 in the int64 version).
//
// (2) Replacing 8 per-step min_tau branches with 1 (batch minimum).
//
//     After computing 8 deltas in SIMD, form an inclusive prefix sum in
//     scalar (7 adds, no branches).  These prefix sums are tau values
//     RELATIVE to the batch start.  A SIMD horizontal min (_mm_min_epi32)
//     over all 8 finds the batch minimum, then ONE comparison updates
//     min_tau.  This removes 7 of the 8 hard-to-predict conditional
//     branches that stalled the original loop.
// ─────────────────────────────────────────────────────────────────────────────
static i64 process_chunk(i64 L, i64 C, i64 e0,
                          const int16_t* t0,
                          const int16_t* t1,
                          const int16_t* t2,
                          i64 tau, i64& min_tau)
{
    i64 k = 0;

#ifdef __AVX2__
    if (L >= 8) {
        // C_vec[lane] = C − lane·e₀  for lane = 0..7  (int32_t; see bounds above).
        // _mm256_set_epi32 takes arguments in REVERSE lane order (lane7 first).
        __m256i C_vec = _mm256_set_epi32(
            (int32_t)(C - 7*e0), (int32_t)(C - 6*e0),
            (int32_t)(C - 5*e0), (int32_t)(C - 4*e0),
            (int32_t)(C - 3*e0), (int32_t)(C - 2*e0),
            (int32_t)(C -   e0), (int32_t) C);
        // After each 8-step batch, advance every lane by 8·e₀.
        const __m256i step8 = _mm256_set1_epi32((int32_t)(8 * e0));

        for (; k + 7 < L; k += 8) {
            // Load 8 consecutive int16_t from each table (16 bytes = one __m128i).
            // In-bounds guarantee: caller ensures phase + k + 7 < alpha for each i.
            __m256i v0 = _mm256_cvtepi16_epi32(
                             _mm_loadu_si128((const __m128i*)(t0 + k)));
            __m256i v1 = _mm256_cvtepi16_epi32(
                             _mm_loadu_si128((const __m128i*)(t1 + k)));
            __m256i v2 = _mm256_cvtepi16_epi32(
                             _mm_loadu_si128((const __m128i*)(t2 + k)));

            // delta_vec[lane] = C_vec[lane] − v0[lane] − v1[lane] − v2[lane]
            __m256i dv = _mm256_sub_epi32(C_vec,
                             _mm256_add_epi32(v0, _mm256_add_epi32(v1, v2)));

            // Store deltas to an aligned stack buffer.
            alignas(32) int32_t d[8];
            _mm256_store_si256(reinterpret_cast<__m256i*>(d), dv);

            // ── Prefix scan (7 scalar adds, no branches) ─────────────────────
            // After this, d[k] = sum of original d[0..k] = tau[j+k+1] − tau[j].
            d[1] += d[0]; d[2] += d[1]; d[3] += d[2];
            d[4] += d[3]; d[5] += d[4]; d[6] += d[5]; d[7] += d[6];

            // ── Batch minimum via SIMD horizontal min ─────────────────────────
            // Find min(d[0..7]) which equals min(tau[j+1..j+8]) − tau[j].
            __m128i lo = _mm_load_si128((const __m128i*)d);        // d[0..3]
            __m128i hi = _mm_load_si128((const __m128i*)(d + 4));  // d[4..7]
            __m128i m  = _mm_min_epi32(lo, hi);
            m = _mm_min_epi32(m, _mm_shuffle_epi32(m, 0x4E));  // swap 64-bit halves
            m = _mm_min_epi32(m, _mm_shuffle_epi32(m, 0xB1));  // swap 32-bit pairs
            int32_t batch_min_rel = _mm_cvtsi128_si32(m);      // horizontal min

            // ONE branch to update min_tau (down from 8).
            i64 batch_min = tau + (i64)batch_min_rel;
            if (batch_min < min_tau) min_tau = batch_min;

            // Advance tau by the total of all 8 deltas (= d[7] after prefix scan).
            tau += (i64)d[7];

            // Advance C_vec to the next batch of 8 steps.
            C_vec = _mm256_sub_epi32(C_vec, step8);
        }
    }
#endif

    // Scalar tail (also handles the entire loop when AVX2 is unavailable or L < 8).
    for (i64 C_k = C - k * e0; k < L; k++, C_k -= e0) {
        tau += C_k - static_cast<i64>(t0[k])
                   - static_cast<i64>(t1[k])
                   - static_cast<i64>(t2[k]);
        if (tau < min_tau) min_tau = tau;
    }

    return tau;
}

// ─────────────────────────────────────────────────────────────────────────────
// compute_min_tau_fast — same result as compute_min_tau, faster inner loop.
//
// Uses the precomputed CeilTableCache to replace irregular Beatty-sequence
// branches with sequential table lookups, then applies SIMD (AVX2) to compute
// four deltas per cycle inside each chunk.
//
// The outer loop finds the next phase-wrap boundary (the smallest remaining
// steps among the three components), calls process_chunk to handle all steps
// in that region, then advances the phases.  Between wrap boundaries, all
// three table pointers advance sequentially — safe for the 4-at-a-time reads.
//
// tau[j=0] = 0 is captured by the initial min_tau = 0.  All subsequent tau
// values are recorded inside process_chunk.
// ─────────────────────────────────────────────────────────────────────────────
i64 compute_min_tau_fast(const SeifertData& sd, const CeilTableCache& cache) {
    // m_max = max(0, N₀)  where  N₀ = pqr − pq − qr − rp.
    // Karakurt: τ is non-decreasing past N₀, so its global minimum lies in [0, N₀].
    // For Σ(2,3,5), N₀ = −1; we clamp to 0 so the τ(0)=0 record is still taken.
    const i64 N0 = sd.alpha[0] * sd.alpha[1] * sd.alpha[2]
                 - sd.alpha[0] * sd.alpha[1]
                 - sd.alpha[1] * sd.alpha[2]
                 - sd.alpha[2] * sd.alpha[0];
    const i64 m_max = N0 > 0 ? N0 : 0;

    const int16_t* tbl[3];
    for (int i = 0; i < 3; i++)
        tbl[i] = cache.table(sd.omega[i], sd.alpha[i]);

    i64 phase[3]        = {0, 0, 0};
    i64 block_offset[3] = {0, 0, 0};
    i64 j_times_e0 = 0;
    i64 tau         = 0;
    i64 min_tau     = 0;   // captures tau[0] = 0

    for (i64 j = 0; j < m_max; ) {
        // Steps until the next phase wrap (whichever component wraps first).
        i64 steps = m_max - j;
        for (int i = 0; i < 3; i++) {
            i64 rem = sd.alpha[i] - phase[i];
            if (rem < steps) steps = rem;
        }
        // steps >= 1 always (phase[i] < alpha[i], so rem >= 1).

        // C = 1 − j·e0 − Σᵢ block_offset[i]  (constant within this chunk).
        const i64 C = 1 - j_times_e0
                    - block_offset[0] - block_offset[1] - block_offset[2];

        // Process `steps` deltas; records tau[j+1..j+steps] in min_tau.
        tau = process_chunk(steps, C, sd.e0,
                            tbl[0] + phase[0],
                            tbl[1] + phase[1],
                            tbl[2] + phase[2],
                            tau, min_tau);

        j          += steps;
        j_times_e0 += steps * sd.e0;

        // Advance phases; exactly one (or more, if two wrap simultaneously)
        // component(s) will hit their boundary.
        for (int i = 0; i < 3; i++) {
            phase[i] += steps;
            if (phase[i] >= sd.alpha[i]) {
                phase[i] -= sd.alpha[i];
                block_offset[i] += sd.omega[i];
            }
        }
    }
    // tau is now tau[m_max], already compared with min_tau in the last chunk.

    return min_tau;
}

// Overload kept for ABI / call-site convenience.  The `cache` parameter is
// no longer consulted: the batch path now goes through compute_min_tau_semigroup
// (Karakurt-style lattice enumeration), which needs no precomputed tables.
DResult d_invariant(i64 p, i64 q, i64 r, const CeilTableCache& /*cache*/) {
    const SeifertData sd = brieskorn_seifert(p, q, r);
    const i64 ks_term    = compute_ks_term(sd);
    const i64 min_tau    = compute_min_tau_semigroup(p, q, r);
    return { ks_term - 2*min_tau, ks_term, 2*min_tau };
}
