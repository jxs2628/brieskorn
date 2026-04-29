# Makefile for d_invariant
#
# Targets:
#   make              — build everything (d_invariant, test binary, benchmark)
#   make test         — build and run the test suite
#   make benchmark    — build and run the performance benchmark
#   make profile      — build with gprof instrumentation, run a scan, show report
#   make clean        — remove all compiled files

CXX      := g++
# -O3 -march=native : aggressive speed optimisations for the current CPU
# -std=c++17        : needed for structured bindings (auto [p,q,r] = ...)
# -fopenmp          : enables OpenMP parallel-for in d_invariant.cpp
CXXFLAGS := -O3 -march=native -std=c++17 -fopenmp

FLINT_DIR := /home/josefsvoboda/local/flint3
FLINT_INC := -I$(FLINT_DIR)/include
# -Wl,-rpath,... : embed the library path into the binary so it can be run
#                  without setting LD_LIBRARY_PATH
FLINT_LIB := -L$(FLINT_DIR)/lib -Wl,-rpath,$(FLINT_DIR)/lib
LIBS      := -lflint -lgmp

INCLUDES := -Icpp/include
    
# All targets depend on the header.
HEADERS := ./cpp/include/d_invariant_core.h

# ─────────────────────────────────────────────────────────────────────────────
# Default target: build all binaries
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: all
all: d_invariant test_d_invariant benchmark

# Shared object file: compiled once, linked into all three binaries.
d_invariant_core.o: cpp/src/d_invariant_core.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(FLINT_INC) -c $< -o $@

d_invariant: cpp/src/d_invariant.cpp d_invariant_core.o $(HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(FLINT_INC) $(FLINT_LIB) $< d_invariant_core.o $(LIBS) -o $@

test_d_invariant: cpp/src/test_d_invariant.cpp d_invariant_core.o $(HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(FLINT_INC) $(FLINT_LIB) $< d_invariant_core.o $(LIBS) -o $@

benchmark: cpp/src/benchmark.cpp d_invariant_core.o $(HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(FLINT_INC) $(FLINT_LIB) $< d_invariant_core.o $(LIBS) -o $@

# ─────────────────────────────────────────────────────────────────────────────
# Run tests
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: test
test: test_d_invariant
	./test_d_invariant

# ─────────────────────────────────────────────────────────────────────────────
# Run benchmark
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: benchmark_run
benchmark_run: benchmark
	./benchmark

# ─────────────────────────────────────────────────────────────────────────────
# gprof profiling
#
# Builds a separate instrumented binary (d_invariant_pg), runs a moderate scan
# so gprof has something to measure, then displays the top functions.
# The full report is written to profile_report.txt.
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: profile
profile: d_invariant_core_pg.o
	$(CXX) $(CXXFLAGS) -pg $(FLINT_INC) $(FLINT_LIB) \
	    d_invariant.cpp d_invariant_core_pg.o $(LIBS) -o d_invariant_pg
	@echo "Running instrumented scan (N=60)…"
	./d_invariant_pg --scan 60 > /dev/null
	gprof d_invariant_pg gmon.out > profile_report.txt
	@echo ""
	@echo "=== Top 15 functions (from gprof) ==="
	@grep -A 30 "Flat profile" profile_report.txt | head -35
	@echo ""
	@echo "Full report written to profile_report.txt"

d_invariant_core_pg.o: d_invariant_core.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -pg $(FLINT_INC) -c $< -o $@

# ─────────────────────────────────────────────────────────────────────────────
# Clean
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: clean
clean:
	rm -f *.o d_invariant test_d_invariant benchmark d_invariant_pg \
	      gmon.out profile_report.txt
