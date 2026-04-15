// benchmark.cpp
//
// Profiles the d-invariant computation to identify bottlenecks.
//
// Run with:  make benchmark
//
// For each test triple Σ(p, q, r), the program:
//   1. Measures wall-clock time for each sub-function individually.
//   2. Repeats the measurement to get stable averages (small triples are fast).
//   3. Prints a breakdown showing which function dominates.
//
// This lets you see at a glance whether time is spent in:
//   • brieskorn_seifert  (Seifert invariants — a few modular inverses)
//   • compute_ks_term    (FLINT rational arithmetic — O(1) operations)
//   • compute_min_tau    (the main hot loop — O(pqr/2) iterations)

#include "d_invariant_core.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Timing helpers
// ─────────────────────────────────────────────────────────────────────────────

using Clock    = std::chrono::high_resolution_clock;
using Seconds  = std::chrono::duration<double>;
using Nanos    = std::chrono::duration<double, std::nano>;

// Runs `fn` exactly `reps` times and returns the total wall-clock time in ms.
// The result is stored in `out` so the compiler can't optimise the call away.
template <typename Fn, typename Out>
static double time_ms(Fn fn, int reps, Out& out) {
    auto t0 = Clock::now();
    for (int i = 0; i < reps; i++) out = fn();
    auto t1 = Clock::now();
    return Seconds(t1 - t0).count() * 1000.0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Print a formatted row of the results table
// ─────────────────────────────────────────────────────────────────────────────
static void print_row(const std::string& label,
                      double ms_total, double ms_percent,
                      i64 pqr_half, int reps) {
    double per_call = ms_total / reps;
    std::cout << "  " << std::left << std::setw(22) << label
              << std::right
              << std::setw(10) << std::fixed << std::setprecision(3) << per_call << " ms"
              << std::setw(8)  << std::fixed << std::setprecision(1) << ms_percent << "%"
              << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark one triple
// ─────────────────────────────────────────────────────────────────────────────
static void benchmark_triple(i64 p, i64 q, i64 r) {
    const i64 pqr      = p * q * r;
    const i64 pqr_half = pqr / 2;

    // Choose repetition count so total time is ~100 ms for stable measurement.
    // Large triples are slow; small ones need many reps.
    int reps = 1;
    if      (pqr_half <      1000) reps = 5000;
    else if (pqr_half <     10000) reps = 500;
    else if (pqr_half <    100000) reps = 50;
    else if (pqr_half <   1000000) reps = 5;

    // ── Time each sub-function ────────────────────────────────────────────────
    //
    // brieskorn_seifert: computes e0 and omega from p,q,r.
    SeifertData sd_out{};
    double ms_seifert = time_ms([&]{ return brieskorn_seifert(p, q, r); }, reps, sd_out);

    // compute_ks_term: computes (K²+s)/4 from the Seifert data.
    // We use the Seifert data computed above.
    i64 ks_out{};
    double ms_ks = time_ms([&]{ return compute_ks_term(sd_out); }, reps, ks_out);

    // compute_min_tau: the main loop — O(pqr/2) iterations.
    i64 tau_out{};
    double ms_tau = time_ms([&]{ return compute_min_tau(sd_out); }, reps, tau_out);

    // Full call (should be ≈ sum of the above, plus very small overhead).
    DResult res_out{};
    double ms_total_fn = time_ms([&]{ return d_invariant(p, q, r); }, reps, res_out);

    // ── Print the breakdown ───────────────────────────────────────────────────
    double ms_per_seifert = ms_seifert / reps;
    double ms_per_ks      = ms_ks      / reps;
    double ms_per_tau     = ms_tau     / reps;
    double ms_total       = ms_per_seifert + ms_per_ks + ms_per_tau;

    // Guard against zero division for trivially small triples.
    double pct_seifert = (ms_total > 0) ? 100.0 * ms_per_seifert / ms_total : 0.0;
    double pct_ks      = (ms_total > 0) ? 100.0 * ms_per_ks      / ms_total : 0.0;
    double pct_tau     = (ms_total > 0) ? 100.0 * ms_per_tau     / ms_total : 0.0;

    // Format pqr/2 with thousands separator for readability.
    std::string pqr_str = std::to_string(pqr_half);
    // Insert commas every 3 digits from the right.
    for (int pos = (int)pqr_str.size() - 3; pos > 0; pos -= 3)
        pqr_str.insert(pos, ",");

    std::cout << "\nΣ(" << p << "," << q << "," << r << ")"
              << "   pqr/2 = " << pqr_str
              << "   d = " << res_out.d
              << "   (averaged over " << reps << " calls)\n";
    std::cout << "  " << std::left  << std::setw(22) << "function"
              << std::right << std::setw(12) << "time/call"
              << std::setw(9) << "share\n";
    std::cout << "  " << std::string(42, '-') << "\n";

    print_row("brieskorn_seifert",  ms_seifert, pct_seifert, pqr_half, reps);
    print_row("compute_ks_term",    ms_ks,      pct_ks,      pqr_half, reps);
    print_row("compute_min_tau",    ms_tau,      pct_tau,     pqr_half, reps);
    std::cout << "  " << std::string(42, '-') << "\n";
    std::cout << "  " << std::left  << std::setw(22) << "total (sum above)"
              << std::right
              << std::setw(10) << std::fixed << std::setprecision(3) << ms_total << " ms\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    std::cout << "=== d_invariant benchmark ===\n";
    std::cout << "Timing each sub-function for representative Σ(p,q,r).\n";
    std::cout << "Triples are ordered by pqr/2 (the loop count in compute_min_tau).\n";

    // Triples span a wide range of sizes.
    // All are prime triples (pairwise coprime by construction).
    struct Triple { i64 p, q, r; };
    std::vector<Triple> triples = {
        {  2,   3,   5},   // pqr/2 =           15
        {  2,   3,  11},   // pqr/2 =           33
        {  2,   5,   7},   // pqr/2 =           35
        {  5,   7,  11},   // pqr/2 =          192
        { 11,  13,  17},   // pqr/2 =        1,214
        { 23,  29,  31},   // pqr/2 =       10,352
        { 47,  53,  59},   // pqr/2 =       73,669
        { 97, 101, 103},   // pqr/2 =      505,453
        {197, 199, 211},   // pqr/2 =    8,264,047
    };

    for (auto& t : triples)
        benchmark_triple(t.p, t.q, t.r);

    // ── Parallel scaling benchmark ────────────────────────────────────────────
    // Use a large triple and show how time scales with thread count.
    {
        const i64 p = 197, q = 199, r = 211;   // pqr/2 ≈ 4.1 M
        SeifertData sd = brieskorn_seifert(p, q, r);

#ifdef _OPENMP
        const int max_threads = omp_get_max_threads();
#else
        const int max_threads = 1;
#endif

        std::cout << "\n── Parallel scaling: Σ(197,199,211)  [pqr/2 ≈ 4.1M] ──\n";
        std::cout << "  " << std::left  << std::setw(10) << "threads"
                  << std::right << std::setw(12) << "time (ms)"
                  << std::setw(12) << "speedup\n";
        std::cout << "  " << std::string(33, '-') << "\n";

        double base_ms = -1.0;
        for (int T = 1; T <= max_threads; T *= 2) {
            i64 result{};
            double ms = time_ms([&]{ return compute_min_tau_parallel(sd, T); }, 3, result);
            ms /= 3.0;
            if (base_ms < 0) base_ms = ms;

            std::cout << "  " << std::left  << std::setw(10) << T
                      << std::right
                      << std::setw(10) << std::fixed << std::setprecision(1) << ms << " ms"
                      << std::setw(9)  << std::fixed << std::setprecision(2) << (base_ms/ms) << "x\n";
        }
        // Verify result matches single-threaded
        i64 ref = compute_min_tau(sd);
        std::cout << "  Result correct: " << (compute_min_tau_parallel(sd, max_threads) == ref ? "yes" : "NO!") << "\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}
