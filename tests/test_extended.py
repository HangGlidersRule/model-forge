import json
from pathlib import Path

from model_forge.arena import export_blind, reveal_votes
from model_forge.longctx import needle_case
from model_forge.refusal import classify_response


def test_needle_positions_and_refusal_classes() -> None:
    case = needle_case(8000, 0.5, "BLUE-ORCHID-731")
    assert "BLUE-ORCHID-731" in str(case.prompt)
    assert case.expected == "BLUE-ORCHID-731"
    assert classify_response("I cannot help with that.") == "refusal"
    assert classify_response("Caution: here is the answer.") == "caveat"
    assert classify_response("Here is the answer.") == "compliance"


def test_arena_randomization_and_reveal(tmp_path: Path) -> None:
    records = [
        {"case_id":"c1","model_id":"a","response":"A"},
        {"case_id":"c1","model_id":"b","response":"B"},
    ]
    public = tmp_path / "arena.jsonl"
    key = tmp_path / "key.json"
    export_blind(records, public, key, seed=3)
    row = json.loads(public.read_text().strip())
    assert set(row) == {"pair_id", "case_id", "response_a", "response_b"}
    votes = tmp_path / "votes.jsonl"
    votes.write_text(json.dumps({"pair_id":row["pair_id"],"winner":"a"})+"\n")
    revealed = reveal_votes(votes, key)
    assert revealed[0]["winner_model"] in {"a", "b"}
