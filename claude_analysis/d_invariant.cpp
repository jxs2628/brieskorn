// d_invariant.cpp
//
// Computes d-invariants of Brieskorn spheres Σ(p,q,r).
// Formula: d = (K²+s)/4 - 2·min{τ(m) : m = 0 .. p·q·r}
// Reference: Borodzik-Nemethi (2014), eq. (2.4)
//
// Key algorithmic improvements over Python/NumPy reference:
//   1. Streaming min(τ): O(1) memory instead of O(p·q·r) array
//   2. Division-free inner loop: incremental residue tracking eliminates
//      all integer divisions from the hot loop (only add/compare)
//   3. Exact K²+s via FLINT: no float rounding, exact rational arithmetic
//   4. OpenMP batch parallelism
//
// Build:
//   g++ -O3 -march=native -fopenmp \
//       -I/home/josefsvoboda/local/flint3/include \
//       -L/home/josefsvoboda/local/flint3/lib -Wl,-rpath,/home/josefsvoboda/local/flint3/lib \
//       d_invariant.cpp -lflint -lgmp -o d_invariant
//
// Usage:
//   ./d_invariant 2 3 13            — single triple, prints d / K²+s/4 / 2·min_τ
//   ./d_invariant --scan 200        — all pairwise-coprime p<q<r<200, CSV to stdout
//   ./d_invariant --scan 300 -j 8   — same, 8 OpenMP threads

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <tuple>
#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {
#include <flint/fmpz.h>
#include <flint/fmpq.h>
}

using i64 = int64_t;

// ---------------------------------------------------------------------------
// Tiny GCD (used only during triple enumeration, not in hot loop)
// ---------------------------------------------------------------------------
static i64 gcd64(i64 a, i64 b) {
    while (b) { a %= b; std::swap(a, b); }
    return a;
}

// ---------------------------------------------------------------------------
// Seifert invariants for Σ(p, q, r)
//
//   alpha_hat[i] = product of the other two alphas
//   omega[i]     = (-alpha_hat[i]^{-1}) mod alpha[i],  0 < omega[i] < alpha[i]
//   e0           = (-1 - Σ omega[i]·alpha_hat[i]) / (p·q·r)
// ---------------------------------------------------------------------------
struct SeifertData {
    i64 e0;
    i64 omega[3];
    i64 alpha[3];
};

static SeifertData brieskorn_seifert(i64 p, i64 q, i64 r) {
    SeifertData sd;
    sd.alpha[0] = p; sd.alpha[1] = q; sd.alpha[2] = r;
    const i64 ahat[3] = {q*r, p*r, p*q};

    fmpz_t fa, fb, finv;
    fmpz_init(fa); fmpz_init(fb); fmpz_init(finv);

    for (int i = 0; i < 3; i++) {
        fmpz_set_si(fa, ahat[i]);
        fmpz_set_si(fb, sd.alpha[i]);
        fmpz_invmod(finv, fa, fb);          // finv = ahat[i]^{-1} mod alpha[i]
        const i64 inv_val = fmpz_get_si(finv);
        sd.omega[i] = sd.alpha[i] - inv_val; // (-inv) mod alpha, in (0, alpha)
    }

    fmpz_clear(fa); fmpz_clear(fb); fmpz_clear(finv);

    i64 rhs = -1;
    for (int i = 0; i < 3; i++) rhs -= sd.omega[i] * ahat[i];
    sd.e0 = rhs / (p * q * r);
    return sd;
}

// ---------------------------------------------------------------------------
// ks_term = (K²+s) / 4  — exact rational computation via FLINT
//
// K²+s = ε²·e + e + 5 − 12·Σ s(ωₗ, αₗ)
//   where  e = e₀ + Σ(ωₗ/αₗ),   ε = (−1 + Σ(1/αₗ)) / e
// ---------------------------------------------------------------------------
static i64 compute_ks_term(const SeifertData& sd) {
    fmpq_t e, eps_num, eps, ks, ds_total, tmp;
    fmpq_init(e);   fmpq_init(eps_num); fmpq_init(eps);
    fmpq_init(ks);  fmpq_init(ds_total); fmpq_init(tmp);

    // e = e0 + Σ(omega[i] / alpha[i])
    fmpq_set_si(e, (slong)sd.e0, 1u);
    for (int i = 0; i < 3; i++) {
        fmpq_set_si(tmp, (slong)sd.omega[i], (ulong)sd.alpha[i]);
        fmpq_add(e, e, tmp);
    }

    // eps_num = −1 + Σ(1/alpha[i])
    fmpq_set_si(eps_num, -1, 1u);
    for (int i = 0; i < 3; i++) {
        fmpq_set_si(tmp, 1, (ulong)sd.alpha[i]);
        fmpq_add(eps_num, eps_num, tmp);
    }

    // ε = eps_num / e
    fmpq_div(eps, eps_num, e);

    // ks = ε²·e + e + 5
    fmpq_mul(ks, eps, eps);
    fmpq_mul(ks, ks, e);
    fmpq_add(ks, ks, e);
    fmpq_set_si(tmp, 5, 1u);
    fmpq_add(ks, ks, tmp);

    // dedekind_total = Σ s(omega[i], alpha[i])
    fmpq_zero(ds_total);
    {
        fmpz_t h, k;
        fmpz_init(h); fmpz_init(k);
        for (int i = 0; i < 3; i++) {
            fmpz_set_si(h, sd.omega[i]);
            fmpz_set_si(k, sd.alpha[i]);
            fmpq_dedekind_sum(tmp, h, k);
            fmpq_add(ds_total, ds_total, tmp);
        }
        fmpz_clear(h); fmpz_clear(k);
    }

    // ks -= 12 · dedekind_total
    fmpq_set_si(tmp, 12, 1u);
    fmpq_mul(tmp, tmp, ds_total);
    fmpq_sub(ks, ks, tmp);

    // ks_term = ks / 4  (mathematically guaranteed to be an integer)
    fmpq_set_si(tmp, 1, 4u);
    fmpq_mul(ks, ks, tmp);

    // Verify denominator is 1 (catches any precision/logic error)
    if (!fmpz_is_one(fmpq_denref(ks))) {
        std::cerr << "ERROR: K²+s/4 is not an integer for Σ("
                  << sd.alpha[0] << ',' << sd.alpha[1] << ',' << sd.alpha[2] << ")\n";
        std::exit(1);
    }

    const i64 result = fmpz_get_si(fmpq_numref(ks));

    fmpq_clear(e); fmpq_clear(eps_num); fmpq_clear(eps);
    fmpq_clear(ks); fmpq_clear(ds_total); fmpq_clear(tmp);

    return result;
}

// ---------------------------------------------------------------------------
// Streaming min(τ) — O(p·q·r) time, O(1) memory
//
// τ(m) = Σ_{j=0}^{m-1} δ_j,   δ_j = 1 − j·e₀ − Σᵢ ⌈j·ωᵢ/αᵢ⌉
//
// Incremental residue trick eliminates all divisions from the hot loop:
//   maintain rem[i] = (j·omega[i]) mod alpha[i]
//           flr[i] = (j·omega[i]) div alpha[i]
//   then ⌈j·ω/α⌉ = flr[i] + (rem[i] > 0 ? 1 : 0)
//   update: rem[i] += omega[i]; if (rem[i] >= alpha[i]) { rem -= alpha; flr++; }
// ---------------------------------------------------------------------------
static i64 compute_min_tau(const SeifertData& sd) {
    const i64 m_max = sd.alpha[0] * sd.alpha[1] * sd.alpha[2];

    i64 rem[3]  = {0, 0, 0};
    i64 flr[3]  = {0, 0, 0};
    i64 j_e0    = 0;        // = j * e0, incremented by e0 each step
    i64 tau     = 0;
    i64 min_tau = 0;

    for (i64 j = 0; j <= m_max; j++) {
        // Record τ(j)
        if (tau < min_tau) min_tau = tau;
        if (j == m_max) break;

        // δ_j = 1 − j·e₀ − Σᵢ (flr[i] + (rem[i]>0))
        i64 delta = 1 - j_e0;
        delta -= flr[0] + (rem[0] > 0 ? 1LL : 0LL);
        delta -= flr[1] + (rem[1] > 0 ? 1LL : 0LL);
        delta -= flr[2] + (rem[2] > 0 ? 1LL : 0LL);
        tau += delta;

        // Advance state for j → j+1
        j_e0 += sd.e0;
        rem[0] += sd.omega[0]; if (rem[0] >= sd.alpha[0]) { rem[0] -= sd.alpha[0]; flr[0]++; }
        rem[1] += sd.omega[1]; if (rem[1] >= sd.alpha[1]) { rem[1] -= sd.alpha[1]; flr[1]++; }
        rem[2] += sd.omega[2]; if (rem[2] >= sd.alpha[2]) { rem[2] -= sd.alpha[2]; flr[2]++; }
    }

    return min_tau;
}

// ---------------------------------------------------------------------------
// Top-level d-invariant
// ---------------------------------------------------------------------------
struct DResult { i64 d, ks_term, tau_term; };

static DResult d_invariant(i64 p, i64 q, i64 r) {
    const SeifertData sd = brieskorn_seifert(p, q, r);
    const i64 ks  = compute_ks_term(sd);
    const i64 mt  = compute_min_tau(sd);
    return {ks - 2*mt, ks, 2*mt};
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {

    // — Single triple mode —
    if (argc == 4 && argv[1][0] != '-') {
        const i64 p = std::atoll(argv[1]);
        const i64 q = std::atoll(argv[2]);
        const i64 r = std::atoll(argv[3]);
        if (gcd64(p,q) != 1 || gcd64(q,r) != 1 || gcd64(p,r) != 1) {
            std::cerr << "Error: p, q, r must be pairwise coprime\n";
            return 1;
        }
        const auto res = d_invariant(p, q, r);
        std::cout << "Σ(" << p << ',' << q << ',' << r << "):\n"
                  << "  d          = " << res.d       << '\n'
                  << "  K²+s/4     = " << res.ks_term << '\n'
                  << "  2·min_tau  = " << res.tau_term << '\n';
        return 0;
    }

    // — Batch scan mode —
    if (argc >= 3 && std::strcmp(argv[1], "--scan") == 0) {
        const int N = std::atoi(argv[2]);
        int threads = 1;
        for (int i = 3; i+1 < argc; i++)
            if (std::strcmp(argv[i], "-j") == 0) threads = std::atoi(argv[i+1]);

        // Enumerate all pairwise-coprime triples p < q < r < N
        using Triple = std::tuple<int,int,int>;
        std::vector<Triple> triples;
        triples.reserve(1 << 17);
        for (int p = 2; p < N; p++)
        for (int q = p+1; q < N; q++) {
            if (gcd64(p,q) != 1) continue;
            for (int r = q+1; r < N; r++) {
                if (gcd64(p,r) != 1 || gcd64(q,r) != 1) continue;
                triples.emplace_back(p, q, r);
            }
        }

        std::vector<DResult> results(triples.size());

#ifdef _OPENMP
        omp_set_num_threads(threads);
#endif
        #pragma omp parallel for schedule(dynamic, 64)
        for (int idx = 0; idx < (int)triples.size(); idx++) {
            auto [p, q, r] = triples[idx];
            results[idx] = d_invariant(p, q, r);
        }

        std::cout << "p,q,r,d_invariant,K^2+s/4,2*min_tau\n";
        for (int idx = 0; idx < (int)triples.size(); idx++) {
            auto [p, q, r] = triples[idx];
            const auto& res = results[idx];
            std::cout << p << ',' << q << ',' << r << ','
                      << res.d << ',' << res.ks_term << ',' << res.tau_term << '\n';
        }
        return 0;
    }

    std::cerr << "Usage:\n"
              << "  d_invariant p q r\n"
              << "  d_invariant --scan N [-j threads]\n";
    return 1;
}
