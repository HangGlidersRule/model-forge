from model_forge.scorecard import Gate, scorecard


def test_scorecard_preserves_categories_and_hard_gates() -> None:
    card = scorecard(
        {"intelligence": 0.8, "instruction": 0.9, "speed": 1.2},
        weights={"intelligence": 0.5, "instruction": 0.3, "speed": 0.2},
        gates=[Gate("instruction", 0.85), Gate("intelligence", 0.75)],
    )
    assert card.hard_gates_passed
    assert card.categories["speed"] == 1.2
    assert card.weighted_score == 0.91
