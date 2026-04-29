// d_invariant_core.h
//
// Declarations for the core d-invariant computation functions.
// Include this header in any file that needs to call the computation routines.
//
// The implementation lives in d_invariant_core.cpp.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

// Shorthand: i64 = 64-bit signed integer.
using i64 = int64_t;

// ─────────────────────────────────────────────────────────────────────────────
// GCD
// ─────────────────────────────────────────────────────────────────────────────
i64 gcd64(i64 a, i64 b);

// ─────────────────────────────────────────────────────────────────────────────
// Seifert invariants for Σ(p, q, r)
//
// Every Seifert-fibered integer homology sphere is encoded by:
//   e0       — integer part of the orbifold Euler number
//   alpha[i] — orders of the three exceptional fibers (= p, q, r)
//   omega[i] — twisting at each fiber; integer in (0, alpha[i])
// ─────────────────────────────────────────────────────────────────────────────
struct SeifertData {
    i64 e0;
    i64 omega[3];
    i64 alpha[3];
};

SeifertData brieskorn_seifert(i64 p, i64 q, i64 r);

// ─────────────────────────────────────────────────────────────────────────────
// Sub-computations (exposed individually for testing and benchmarking)
// ─────────────────────────────────────────────────────────────────────────────

// Returns (K² + s) / 4 as an exact integer using FLINT rational arithmetic.
i64 compute_ks_term(const SeifertData& sd);

// Returns  min{ τ(m) : m = 0, ..., N₀ }   where N₀ = pqr − pq − qr − pr.
// (Karakurt: τ is non-decreasing past N₀, so the global minimum lies in [0, N₀].
//  For Σ(2,3,5), N₀ = −1; we clamp to 0 so τ(0) = 0 is taken.)
i64 compute_min_tau(const SeifertData& sd);

// Returns the full sequence τ(0), τ(1), …, τ(N₀).
// For inspection / plotting; not used by the d-invariant computation itself.
std::vector<i64> compute_tau_sequence(const SeifertData& sd);

// Returns min τ via Karakurt's Theorem 1.3 — enumerates the numerical
// semigroup G(pq, pr, qr) ∩ [0, N₀] (N₀ = pqr − pq − qr − pr) and merges
// up/down events to find min τ in O(|G ∩ [0, N₀]|) time.  Falls back to
// compute_min_tau for (2,3,5) where N₀ < 0.
i64 compute_min_tau_semigroup(i64 p, i64 q, i64 r);

// ─────────────────────────────────────────────────────────────────────────────
// floor_sum and tau_at — O(log α) random access to τ
//
// floor_sum(n, h, k) = Σⱼ₌₀ⁿ⁻¹ ⌊j·h/k⌋
//   Runs in O(log k) steps via a lattice-point-swap recursion that is
//   structurally identical to the Euclidean algorithm.
//
// tau_at(m, sd) = τ(m) computed directly from the closed-form formula
//   τ(m) = m − e0·m(m−1)/2 − Σᵢ [ floor_sum(m,ωᵢ,αᵢ) + (m−1) − ⌊(m−1)/αᵢ⌋ ]
//   This is exact algebra — a rearrangement of the incremental definition.
//   It lets any thread "jump into" the τ sequence at an arbitrary position.
// ─────────────────────────────────────────────────────────────────────────────
i64 floor_sum(i64 n, i64 h, i64 k);
i64 tau_at(i64 m, const SeifertData& sd);

// Parallel version of compute_min_tau.
// Splits [0, N₀] into num_threads equal chunks.  Each thread computes
// τ at its starting position via tau_at (O(log α)), then runs the same
// incremental loop as compute_min_tau over its slice.
// With num_threads = 1 the result is identical to compute_min_tau.
i64 compute_min_tau_parallel(const SeifertData& sd, int num_threads);

// ─────────────────────────────────────────────────────────────────────────────
// CeilTableCache — precomputed ⌈k·ω/α⌉ tables for all (ω,α) with α ≤ N.
//
// For each pair (ω, α) with 1 ≤ ω < α ≤ N, stores α int16_t values:
//   table(ω, α)[k] = ⌈k·ω/α⌉   for k = 0, …, α−1.
//
// Storage layout: a single flat vector.  offsets[α] is the start of α's block;
// within that block, ω occupies the range [(ω−1)·α, ω·α).
//
// Use build_ceil_tables(N) to populate the cache before a batch scan.
// For N = 700 this uses ≈ 228 MB and takes < 1 second.
// ─────────────────────────────────────────────────────────────────────────────
struct CeilTableCache {
    std::vector<int16_t> data;    // flat storage: all ceil tables
    std::vector<size_t>  offsets; // offsets[alpha] = start of alpha's block in data

    // Returns a pointer to alpha int16_t entries:  result[k] = ⌈k·omega/alpha⌉.
    const int16_t* table(i64 omega, i64 alpha) const noexcept {
        return data.data() + offsets[alpha] + (size_t)(omega - 1) * (size_t)alpha;
    }

    bool empty() const noexcept { return data.empty(); }
};

// Build CeilTableCache for all (omega, alpha) with 1 ≤ omega < alpha ≤ N.
// Typically called once before a batch scan; safe to read concurrently.
CeilTableCache build_ceil_tables(int N);

// ─────────────────────────────────────────────────────────────────────────────
// Top-level result
// ─────────────────────────────────────────────────────────────────────────────
struct DResult {
    i64 d;          // the d-invariant:  d = ks_term − tau_term
    i64 ks_term;    // (K² + s) / 4
    i64 tau_term;   // 2 · min τ
};

DResult d_invariant(i64 p, i64 q, i64 r);

// Faster variant: uses the precomputed period tables to eliminate the irregular
// conditional branches from the inner τ scan loop.
i64     compute_min_tau_fast(const SeifertData& sd, const CeilTableCache& cache);
DResult d_invariant(i64 p, i64 q, i64 r, const CeilTableCache& cache);
