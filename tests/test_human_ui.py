from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "eval" / "human.html").read_text(encoding="utf-8")


def test_score_buttons_write_the_persisted_fields():
    """Regression: buttons used short keys, saveNow read human_* fields."""
    assert "item.human_completeness = patch.completeness" in HTML
    assert "item.human_relevance = patch.relevance" in HTML
    assert "item.human_refused = patch.refused" in HTML


def test_finish_is_gated_on_full_progress():
    assert "finish.disabled = state.progress.done < state.progress.total" in HTML


def test_progress_update_is_wired_after_save():
    assert "state.progress = d.progress" in HTML
    assert "applyProgressOnly()" in HTML
    assert 'id="bar"' in HTML
    assert 'id="progressText"' in HTML


def test_blind_scoring_fields_are_not_rendered():
    # The API omits them; the page should never depend on them either.
    assert "judge_completeness" not in HTML
    assert "human_check_key.csv" not in HTML
