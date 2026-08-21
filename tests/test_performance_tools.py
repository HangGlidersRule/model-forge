from model_forge.performance import PerformanceCell, build_performance_cases
from model_forge.tools import score_tool_call


def test_performance_matrix_has_all_dimensions() -> None:
    cells = build_performance_cases(
        concurrencies=[1, 2, 4, 8], prompt_tokens=[128, 4096, 16384, 48000], output_tokens=[32, 512]
    )
    assert len(cells) == 32
    assert PerformanceCell(8, 48000, 512) in cells


def test_nested_and_parallel_tool_calls() -> None:
    expected = [
        {"name":"weather","arguments":{"place":{"city":"DC"},"units":"f"}},
        {"name":"time","arguments":{"zone":"America/New_York"}},
    ]
    actual = [
        {"function":{"name":"time","arguments":"{\"zone\":\"America/New_York\"}"}},
        {"function":{"name":"weather","arguments":"{\"place\":{\"city\":\"DC\"},\"units\":\"f\"}"}},
    ]
    assert score_tool_call(expected, actual).passed
