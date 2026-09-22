from pathlib import Path
import importlib.util


MODULE_PATH = Path(__file__).parents[1] / "src" / "analyze.py"
SPEC = importlib.util.spec_from_file_location("heart_wall_analyze", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_solid_rectangular_wall() -> None:
    metrics = MODULE.analyze_text("💛\n💛💛💛\n💛💛💛")
    assert metrics["opening_family"] == "yellow"
    assert metrics["wall_rows"] == 2
    assert metrics["wall_total_hearts"] == 6
    assert metrics["wall_rectangular"] is True
    assert metrics["wall_distinct_families"] == 1
    assert metrics["barber_pole_exact"] is False


def test_checkerboard_is_detected() -> None:
    metrics = MODULE.analyze_text("💛\n💛💙💛💙\n💙💛💙💛\n💛💙💛💙")
    assert metrics["checkerboard_exact"] is True
    assert metrics["wall_family_entropy_bits"] == 1.0
    assert metrics["wall_rectangular"] is True


def test_diagonal_periodic_wall_is_detected() -> None:
    metrics = MODULE.analyze_text("❤️\n❤️💛💙❤️\n💙❤️💛💙\n💛💙❤️💛")
    assert metrics["barber_pole_exact"] is True
    assert metrics["periodic_fit"] == 1.0
    assert metrics["periodic_slope"] in {-1, 1}


def test_spaces_are_separate_compliance_failure() -> None:
    metrics = MODULE.analyze_text("💚\n  💚💚\n  💚💚")
    assert metrics["wall_total_hearts"] == 4
    assert metrics["wall_parseable"] is False
    assert metrics["leading_indent_rows"] == 2
    assert metrics["wall_only_hearts_and_linebreaks"] is False


def test_missing_response() -> None:
    assert MODULE.analyze_text(None) == {"has_response": False}


def test_bootstrap_interval_is_deterministic_and_contains_mean() -> None:
    first = MODULE.bootstrap_mean_ci95([10.0, 20.0, 30.0], "fixed", draws=500)
    second = MODULE.bootstrap_mean_ci95([10.0, 20.0, 30.0], "fixed", draws=500)
    assert first == second
    assert first[0] <= 20.0 <= first[1]
