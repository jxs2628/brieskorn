// d_invariant.cpp  —  command-line front end
//
// Computes the Heegaard Floer d-invariant of a Brieskorn sphere Σ(p, q, r).
//
// A Brieskorn sphere Σ(p, q, r) is the smooth 3-manifold defined as the
// intersection of the complex surface  z_1^p + z_2^q + z_3^r = 0  with the
// unit sphere in C^3.  The parameters p, q, r must be pairwise coprime
// integers ≥ 2 for Σ(p,q,r) to be an integer homology sphere.
//
// The d-invariant (also called the correction term) is computed via:
//
//   d(Σ(p,q,r)) = (K² + s) / 4  −  2 · min{ τ(m) : m = 0, 1, ..., N₀ }
//   where N₀ = pqr − pq − qr − pr (the Frobenius bound from Karakurt Thm 1.3).
//
// Reference: Borodzik–Nemethi (2014), equation (2.4).
//
// The actual computation is in d_invariant_core.cpp.
// This file only contains the command-line interface (main).
//
// ── Build ────────────────────────────────────────────────────────────────────
//   g++ -O3 -march=native -fopenmp \
//       -I/home/josefsvoboda/local/flint3/include \
//       -L/home/josefsvoboda/local/flint3/lib \
//       -Wl,-rpath,/home/josefsvoboda/local/flint3/lib \
//       d_invariant_core.cpp d_invariant.cpp -lflint -lgmp -o d_invariant
//
//   Or simply:  make
//
// ── Usage ────────────────────────────────────────────────────────────────────
//   ./d_invariant 2 3 13            — single triple, prints d / K²+s/4 / 2·min_τ
//   ./d_invariant 2 3 13 --tau      — same, plus full τ sequence as CSV
//   ./d_invariant --scan 200        — all pairwise-coprime p<q<r<200, CSV to stdout
//   ./d_invariant --scan 300 -j 8   — same, using 8 OpenMP threads

#include "d_invariant_core.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <tuple>
#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char* argv[]) {

    // ── Mode 1: single triple ─────────────────────────────────────────────────
    if (argc >= 4 && argv[1][0] != '-' && argv[2][0] != '-' && argv[3][0] != '-') {
        const i64 p = std::atoll(argv[1]);
        const i64 q = std::atoll(argv[2]);
        const i64 r = std::atoll(argv[3]);

        // Optional flags after the triple:
        //   --tau    print the full τ sequence as CSV (m,tau) to stdout
        bool print_tau = false;
        for (int i = 4; i < argc; i++) {
            if (std::strcmp(argv[i], "--tau") == 0) print_tau = true;
        }

        if (gcd64(p,q) != 1 || gcd64(q,r) != 1 || gcd64(p,r) != 1) {
            std::cerr << "Error: p, q, r must be pairwise coprime\n";
            return 1;
        }

        const auto res = d_invariant(p, q, r);
        std::cout << "Σ(" << p << ',' << q << ',' << r << "):\n"
                  << "  d         = " << res.d        << '\n'
                  << " k^2+s      = " << res.ks_term << '\n'
                  << "N_0         = " << p*q*r-p*q-q*r-r*p << '\n';

        if (print_tau) {
            const SeifertData sd = brieskorn_seifert(p, q, r);
            const std::vector<i64> tau_seq = compute_tau_sequence(sd);
            //std::cout << "tau\n";
            //for (size_t m = 0; m < tau_seq.size(); m++) {
            //    std::cout  << tau_seq[m];
            //    if (m + 1 < tau_seq.size()) std::cout << ' ';}
            std::cout << "\n ks-2 tau\n";
            for (size_t m = 0; m < tau_seq.size(); m++) {
                std::cout << res.ks_term- 2* tau_seq[m];
                if (m + 1 < tau_seq.size()) std::cout << ' ';
            }
            std::cout << '\n';
        }
        return 0;
    }

    // ── Mode 2: batch scan ────────────────────────────────────────────────────
    if (argc >= 3 && std::strcmp(argv[1], "--scan") == 0) {
        const int N       = std::atoi(argv[2]);
        int       threads = 1;
        int       rmin    = 0;

        // Parse optional flags:
        //   -j <threads>      OpenMP threads
        //   --rmin <R>        only emit triples with r >= R (resume / range scan)
        for (int i = 3; i+1 < argc; i++) {
            if (std::strcmp(argv[i], "-j") == 0)        threads = std::atoi(argv[i+1]);
            else if (std::strcmp(argv[i], "--rmin") == 0) rmin  = std::atoi(argv[i+1]);
        }

        // Enumerate all pairwise-coprime triples with p < q < r < N
        // and r >= rmin (default 0 = no restriction).
        using Triple = std::tuple<int,int,int>;
        std::vector<Triple> triples;
        triples.reserve(1 << 17);
        for (int p = 2; p < N; p++)
        for (int q = p+1; q < N; q++) {
            if (gcd64(p,q) != 1) continue;
            const int r_start = std::max(q+1, rmin);
            for (int r = r_start; r < N; r++) {
                if (gcd64(p,r) != 1 || gcd64(q,r) != 1) continue;
                triples.emplace_back(p, q, r);
            }
        }

        // LPT-style scheduling: sort by descending p*q*r so the longest jobs
        // start first. Combined with `schedule(dynamic, 64)` this prevents a
        // few near-N triples from leaving cores idle at the tail of the run.
        std::sort(triples.begin(), triples.end(),
                  [](const Triple& a, const Triple& b) {
                      const auto& [pa, qa, ra] = a;
                      const auto& [pb, qb, rb] = b;
                      return (i64)pa*qa*ra > (i64)pb*qb*rb;
                  });

        // The Karakurt-semigroup path inside d_invariant(p,q,r,cache) ignores
        // the cache, so we pass an empty one — saves ~N³ × 2/3 bytes
        // (≈ 228 MB at N=700) and the build pass.
        const CeilTableCache cache;

        // Compute d-invariants, optionally in parallel.
        std::vector<DResult> results(triples.size());

#ifdef _OPENMP
        omp_set_num_threads(threads);
#endif
        #pragma omp parallel for schedule(dynamic, 64)
        for (int idx = 0; idx < (int)triples.size(); idx++) {
            auto [p, q, r] = triples[idx];
            results[idx] = d_invariant(p, q, r, cache);
        }

        // Print results as CSV.
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
              << "  d_invariant p q r [--tau]\n"
              << "    --tau      also print the full τ sequence as CSV (m,tau)\n"
              << "  d_invariant --scan N [-j threads] [--rmin R]\n"
              << "    --rmin R   only emit triples with r >= R (e.g. resume from prior scan)\n";
    return 1;
}
