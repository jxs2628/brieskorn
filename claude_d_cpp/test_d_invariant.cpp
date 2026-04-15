// test_d_invariant.cpp
//
// Tests for the d-invariant computation.
//
// Run with:  make test
//
// The test cases come from:
//   1. Analytically known values (Poincaré sphere, etc.)
//   2. Values pre-verified against the Python reference implementation
//      (all 77,558 triples with p < q < r < 120 were checked).
//
// A test fails if the computed value differs from the expected value.
// The program exits with code 0 if all tests pass, 1 if any fail.

#include "d_invariant_core.h"
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>  // std::min_element

// ─────────────────────────────────────────────────────────────────────────────
// Minimal test framework
// ─────────────────────────────────────────────────────────────────────────────

static int tests_run    = 0;
static int tests_passed = 0;
static int tests_failed = 0;

// Check that 'got' equals 'expected'.  Prints PASS or FAIL.
// We fix T = i64 explicitly to avoid integer-type deduction mismatches
// (on some platforms int64_t is 'long' while literals like 2LL are 'long long').
static void check(const std::string& name, i64 got, i64 expected) {
    tests_run++;
    if (got == expected) {
        tests_passed++;
        std::cout << "  PASS  " << name << "\n";
    } else {
        tests_failed++;
        std::cout << "  FAIL  " << name
                  << "  (got " << got << ", expected " << expected << ")\n";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: gcd64
// ─────────────────────────────────────────────────────────────────────────────
static void test_gcd() {
    std::cout << "\n── gcd64 ─────────────────────────────────────────────────\n";
    check("gcd(6,4)",    gcd64(6,4),     2);
    check("gcd(7,13)",   gcd64(7,13),    1);
    check("gcd(100,75)", gcd64(100,75), 25);
    check("gcd(1,1)",    gcd64(1,1),     1);
    check("gcd(30,42)",  gcd64(30,42),   6);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: brieskorn_seifert — check e0 and omega values
//
// These are computed by hand from the definition:
//   omega[i] = alpha[i] − (alpha_hat[i]^{−1} mod alpha[i])
//   e0 = (−1 − Σ omega[i]·alpha_hat[i]) / (p·q·r)
// ─────────────────────────────────────────────────────────────────────────────
static void test_seifert() {
    std::cout << "\n── brieskorn_seifert ─────────────────────────────────────\n";

    // Σ(2,3,5):  alpha_hat = {15, 10, 6}
    //   omega[0] = 2 − (15^{-1} mod 2) = 2 − 1 = 1
    //   omega[1] = 3 − (10^{-1} mod 3) = 3 − 1 = 2
    //   omega[2] = 5 − (6^{-1}  mod 5) = 5 − 1 = 4
    //   e0 = (−1 − 15 − 20 − 24) / 30 = −60/30 = −2
    {
        SeifertData sd = brieskorn_seifert(2, 3, 5);
        check("Σ(2,3,5) e0",       sd.e0,       -2);
        check("Σ(2,3,5) omega[0]",  sd.omega[0],  1);
        check("Σ(2,3,5) omega[1]",  sd.omega[1],  2);
        check("Σ(2,3,5) omega[2]",  sd.omega[2],  4);
    }

    // Σ(2,3,7)  (Poincaré homology sphere):  alpha_hat = {21, 14, 6}
    //   omega[0] = 2 − (21^{-1} mod 2) = 2 − 1 = 1
    //   omega[1] = 3 − (14^{-1} mod 3) = 3 − 2 = 1   [14 ≡ 2 mod 3, 2^{-1} = 2]
    //   omega[2] = 7 − (6^{-1}  mod 7) = 7 − 6 = 1   [6·6 = 36 ≡ 1 mod 7]
    //   e0 = (−1 − 21 − 14 − 6) / 42 = −42/42 = −1
    {
        SeifertData sd = brieskorn_seifert(2, 3, 7);
        check("Σ(2,3,7) e0",       sd.e0,       -1);
        check("Σ(2,3,7) omega[0]",  sd.omega[0],  1);
        check("Σ(2,3,7) omega[1]",  sd.omega[1],  1);
        check("Σ(2,3,7) omega[2]",  sd.omega[2],  1);
    }

    // Σ(2,3,11):  alpha_hat = {33, 22, 6}
    //   omega[0] = 1  (33 ≡ 1 mod 2)
    //   omega[1] = 2  (22 ≡ 1 mod 3, inv=1)
    //   omega[2] = 9  (6^{-1} mod 11 = 2, since 6·2=12≡1; omega=11−2=9)
    //   e0 = (−1 − 33 − 44 − 54) / 66 = −132/66 = −2
    {
        SeifertData sd = brieskorn_seifert(2, 3, 11);
        check("Σ(2,3,11) e0",       sd.e0,       -2);
        check("Σ(2,3,11) omega[0]",  sd.omega[0],  1);
        check("Σ(2,3,11) omega[1]",  sd.omega[1],  2);
        check("Σ(2,3,11) omega[2]",  sd.omega[2],  9);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: compute_ks_term
// ─────────────────────────────────────────────────────────────────────────────
static void test_ks_term() {
    std::cout << "\n── compute_ks_term ───────────────────────────────────────\n";

    // Values verified against Python reference (using exact Fraction arithmetic).
    struct Case { i64 p, q, r, expected_ks; };
    std::vector<Case> cases = {
        {2, 3,  5,   2},
        {2, 3,  7,   0},
        {2, 3, 11,   2},
        {2, 3, 13,   0},
        {2, 3, 17,   2},
        {2, 3, 19,   0},
        {2, 5,  7,   0},
        {2, 5,  9,   2},
        {3, 5,  7,   0},
        {5, 7, 11, -30},
    };

    for (auto& c : cases) {
        SeifertData sd = brieskorn_seifert(c.p, c.q, c.r);
        i64 ks = compute_ks_term(sd);
        check("Σ(" + std::to_string(c.p) + "," + std::to_string(c.q) + ","
              + std::to_string(c.r) + ") ks_term",
              ks, c.expected_ks);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: compute_min_tau
// ─────────────────────────────────────────────────────────────────────────────
static void test_min_tau() {
    std::cout << "\n── compute_min_tau ───────────────────────────────────────\n";

    // min_tau = (ks_term − d) / 2
    struct Case { i64 p, q, r, expected_min_tau; };
    std::vector<Case> cases = {
        {2, 3,  5,   0},   // d=2,  ks=2,   2*min_tau=0
        {2, 3,  7,   0},   // d=0,  ks=0,   2*min_tau=0
        {2, 3, 11,   0},   // d=2,  ks=2,   2*min_tau=0
        {2, 3, 13,   0},   // d=0,  ks=0,   2*min_tau=0
        {2, 5,  7,   0},   // d=0,  ks=0,   2*min_tau=0
        {2, 5,  9,   0},   // d=2,  ks=2,   2*min_tau=0
        {3, 5,  7,  -1},   // d=2,  ks=0,   2*min_tau=-2  → min_tau=-1
        {5, 7, 11, -16},   // d=2,  ks=-30, 2*min_tau=-32 → min_tau=-16
    };

    for (auto& c : cases) {
        SeifertData sd = brieskorn_seifert(c.p, c.q, c.r);
        i64 mt = compute_min_tau(sd);
        check("Σ(" + std::to_string(c.p) + "," + std::to_string(c.q) + ","
              + std::to_string(c.r) + ") min_tau",
              mt, c.expected_min_tau);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: d_invariant  (the full pipeline)
// ─────────────────────────────────────────────────────────────────────────────
static void test_d_invariant() {
    std::cout << "\n── d_invariant ───────────────────────────────────────────\n";

    // Values from the verified CSV (all 77 558 rows checked against Python).
    struct Case { i64 p, q, r, expected_d; };
    std::vector<Case> cases = {
        // d=0 families
        {2,  3,  7,  0},   // Poincaré sphere
        {2,  3, 13,  0},
        {2,  3, 19,  0},
        {2,  3, 25,  0},
        {2,  5,  7,  0},
        {2,  5,  9,  2},   // d=2, not 0
        {2,  7,  9,  2},   // d=2, not 0
        {3,  5,  7,  2},   // d=2, not 0
        {3,  5, 19,  0},
        {5,  7, 11,  2},   // d=2, not 0
        {5,  7, 13,  2},   // d=2, not 0
        // d=2 family (Σ(2, 3, 6k−1) has d=2 for k odd, d=0 for k even)
        {2,  3,  5,  2},
        {2,  3, 11,  2},
        {2,  3, 17,  2},
        {2,  3, 23,  2},
        // A few larger cases
        {2,  5, 11,  0},   // d=0, not 2
        {2,  7, 11,  2},   // d=2, not 0
        {3,  7, 11,  0},
        {5, 11, 13,  0},
        {7, 11, 13,  0},
    };

    for (auto& c : cases) {
        DResult res = d_invariant(c.p, c.q, c.r);
        check("d(Σ(" + std::to_string(c.p) + "," + std::to_string(c.q) + ","
              + std::to_string(c.r) + "))",
              res.d, c.expected_d);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: floor_sum — spot checks against direct summation
// ─────────────────────────────────────────────────────────────────────────────
static void test_floor_sum() {
    std::cout << "\n── floor_sum ─────────────────────────────────────────────\n";

    // Brute-force reference: compute Σⱼ₌₀ⁿ⁻¹ ⌊j·h/k⌋ by a plain loop.
    auto brute = [](i64 n, i64 h, i64 k) -> i64 {
        i64 s = 0;
        for (i64 j = 0; j < n; j++) s += (j * h) / k;
        return s;
    };

    struct Case { i64 n, h, k; };
    std::vector<Case> cases = {
        {1,  1, 2}, {4,  1, 3}, {7,  2, 3},
        {10, 3, 7}, {20, 5, 9}, {100, 7, 13},
        {42, 1, 7}, {42, 6, 7}, {100, 11, 17},
        {1,  0, 5}, {0, 3, 7},   // edge cases: h=0 and n=0
    };

    for (auto& c : cases) {
        i64 got      = floor_sum(c.n, c.h, c.k);
        i64 expected = brute(c.n, c.h, c.k);
        check("floor_sum(" + std::to_string(c.n) + "," + std::to_string(c.h)
              + "," + std::to_string(c.k) + ")",
              got, expected);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: tau_at — dense comparison against the incremental loop
//
// For each test triple, we run the existing incremental loop to build a
// reference vector of τ(0), τ(1), ..., τ(pqr/2), then check that tau_at
// returns the identical value for every m.
// ─────────────────────────────────────────────────────────────────────────────

// Reference: build full τ vector using the incremental loop (same code as
// compute_min_tau, but stores every value instead of tracking only the minimum).
static std::vector<i64> tau_vector(const SeifertData& sd) {
    const i64 m_max = (sd.alpha[0] * sd.alpha[1] * sd.alpha[2]) / 2;
    std::vector<i64> vals(m_max + 1);

    i64 remainder[3]  = {0, 0, 0};
    i64 floor_part[3] = {0, 0, 0};
    i64 j_times_e0    = 0;
    i64 tau           = 0;

    for (i64 j = 0; j <= m_max; j++) {
        vals[j] = tau;
        if (j == m_max) break;

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
    return vals;
}

static void test_tau_at() {
    std::cout << "\n── tau_at  (vs. incremental reference) ──────────────────\n";

    struct Triple { i64 p, q, r; };
    std::vector<Triple> triples = {
        {2, 3,  5},   // pqr/2 = 15
        {2, 3,  7},   // pqr/2 = 21
        {2, 5,  7},   // pqr/2 = 35
        {3, 5,  7},   // pqr/2 = 52
        {5, 7, 11},   // pqr/2 = 192
    };

    for (auto& t : triples) {
        SeifertData sd  = brieskorn_seifert(t.p, t.q, t.r);
        auto ref        = tau_vector(sd);
        const i64 m_max = (i64)ref.size() - 1;

        int mismatches = 0;
        for (i64 m = 0; m <= m_max; m++) {
            if (tau_at(m, sd) != ref[m]) mismatches++;
        }

        std::string name = "tau_at == incremental for every m in Σ("
                         + std::to_string(t.p) + "," + std::to_string(t.q)
                         + "," + std::to_string(t.r) + ")  [m_max="
                         + std::to_string(m_max) + "]";

        // Report as a single PASS/FAIL; on failure show how many mismatches.
        tests_run++;
        if (mismatches == 0) {
            tests_passed++;
            std::cout << "  PASS  " << name << "\n";
        } else {
            tests_failed++;
            std::cout << "  FAIL  " << name
                      << "  (" << mismatches << " mismatches)\n";
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: compute_min_tau_parallel — matches compute_min_tau for several triples
// ─────────────────────────────────────────────────────────────────────────────
static void test_parallel_min_tau() {
    std::cout << "\n── compute_min_tau_parallel ──────────────────────────────\n";

    struct Triple { i64 p, q, r; };
    std::vector<Triple> triples = {
        {2, 3,  5}, {2, 3,  7}, {2, 5,  7}, {3, 5, 7},
        {5, 7, 11}, {2, 3, 25}, {3, 5, 19}, {7, 11, 13},
    };

    for (auto& t : triples) {
        SeifertData sd     = brieskorn_seifert(t.p, t.q, t.r);
        const i64 expected = compute_min_tau(sd);

        for (int threads : {1, 2, 4}) {
            i64 got = compute_min_tau_parallel(sd, threads);
            check("parallel(" + std::to_string(threads) + " threads) Σ("
                  + std::to_string(t.p) + "," + std::to_string(t.q) + ","
                  + std::to_string(t.r) + ")",
                  got, expected);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: internal consistency  d = ks_term − tau_term
// ─────────────────────────────────────────────────────────────────────────────
static void test_consistency() {
    std::cout << "\n── internal consistency ──────────────────────────────────\n";

    // For any triple, d_invariant must satisfy d = ks_term − tau_term.
    struct Triple { i64 p, q, r; };
    std::vector<Triple> triples = {
        {2,3,5}, {2,3,7}, {2,5,7}, {3,5,7}, {3,5,11}, {5,7,9}, {7,11,13}
    };

    for (auto& t : triples) {
        DResult res = d_invariant(t.p, t.q, t.r);
        check("d = ks − tau for Σ(" + std::to_string(t.p) + ","
              + std::to_string(t.q) + "," + std::to_string(t.r) + ")",
              res.d, res.ks_term - res.tau_term);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    std::cout << "=== d_invariant test suite ===\n";

    test_gcd();
    test_floor_sum();
    test_seifert();
    test_ks_term();
    test_min_tau();
    test_tau_at();
    test_parallel_min_tau();
    test_d_invariant();
    test_consistency();

    std::cout << "\n─────────────────────────────────────────────────────────\n";
    std::cout << "Result: " << tests_passed << "/" << tests_run << " passed";
    if (tests_failed > 0) {
        std::cout << "  (" << tests_failed << " FAILED)\n";
        return 1;
    }
    std::cout << "  — all OK\n";
    return 0;
}
