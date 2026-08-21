from model_forge.cases import Case, stable_case_id
from model_forge.scoring import evaluate


def test_stable_case_id_ignores_dictionary_order() -> None:
    a = stable_case_id("math", {"prompt": "2+2", "expected": {"b": 2, "a": 1}})
    b = stable_case_id("math", {"expected": {"a": 1, "b": 2}, "prompt": "2+2"})
    assert a == b


def test_exact_json_and_instruction_scorers() -> None:
    exact = Case(id="x", suite="intelligence", prompt="2+2", scorer="exact", expected="4")
    assert evaluate(exact, " 4 \n").score == 1.0
    schema = Case(
        id="j",
        suite="structured",
        prompt="json",
        scorer="json_schema",
        expected={"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}},
    )
    assert evaluate(schema, '{"name":"Ada"}').score == 1.0
    assert evaluate(schema, "```json\n{\"name\":\"Ada\"}\n```").score == 0.0
    constraint = Case(
        id="i", suite="instruction", prompt="x", scorer="constraints",
        expected={"required": ["alpha", "omega"], "forbidden": ["sorry"], "max_words": 4, "ordered": True},
    )
    assert evaluate(constraint, "alpha then omega").score == 1.0
    assert evaluate(constraint, "omega alpha sorry").score == 0.0
