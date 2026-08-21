from model_forge.stats import bootstrap_paired_delta, percentiles


def test_percentiles_and_deterministic_paired_bootstrap() -> None:
    assert percentiles([1, 2, 3, 4], [50, 90]) == {"p50": 2.5, "p90": 3.7}
    one = bootstrap_paired_delta([2, 4, 6], [1, 2, 3], seed=7, samples=500)
    two = bootstrap_paired_delta([2, 4, 6], [1, 2, 3], seed=7, samples=500)
    assert one == two
    assert one.estimate == 2.0
    assert one.low <= one.estimate <= one.high
