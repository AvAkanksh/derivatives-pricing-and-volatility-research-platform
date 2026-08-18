import pytest
from dpp.benchmark.bench_pricers import run_performance_benchmarks

def test_performance_benchmarks():
    perf = run_performance_benchmarks()
    # Check that performance timings are positive values
    assert perf["binomial_tree_warmup_ms"] > 0
    assert perf["binomial_tree_steady_us"] > 0
    assert perf["pde_warmup_ms"] > 0
    assert perf["pde_steady_ms"] > 0
