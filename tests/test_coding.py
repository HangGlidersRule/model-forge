from model_forge.coding import run_python_tests


def test_code_runner_passes_and_times_out() -> None:
    passed = run_python_tests("def add(a,b): return a+b", "assert add(2,3)==5", timeout=2)
    assert passed.passed
    timed = run_python_tests("while True: pass", "", timeout=0.1)
    assert not timed.passed and timed.timed_out
