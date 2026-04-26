"""Tests for per-component selectors."""

from levi.selection import (
    RoundRobinComponentSelector,
    StagnationComponentSelector,
    UCBComponentSelector,
    make_component_selector,
)


def test_round_robin_cycles_through_targets():
    sel = RoundRobinComponentSelector()
    targets = ["a", "b", "c"]
    picks = [sel.select(targets) for _ in range(6)]
    assert picks == ["a", "b", "c", "a", "b", "c"]


def test_ucb_warms_up_every_target_before_exploiting():
    sel = UCBComponentSelector(c=2.0)
    targets = ["a", "b", "c"]
    first_three = set()
    for _ in range(3):
        target = sel.select(targets)
        first_three.add(target)
        sel.update(target, accepted=False)
    assert first_three == set(targets)


def test_ucb_concentrates_on_high_success_target():
    sel = UCBComponentSelector(c=0.1)  # low exploration
    targets = ["a", "b"]
    # Warm up: one sample each.
    for t in targets:
        sel.update(t, accepted=(t == "a"))
    # Now "a" has success_rate=1.0, "b"=0.0.
    for _ in range(20):
        t = sel.select(targets)
        sel.update(t, accepted=(t == "a"))
    stats = sel.stats()
    assert stats["a"]["n_samples"] > stats["b"]["n_samples"]


def test_stagnation_picks_low_success_target():
    sel = StagnationComponentSelector(c=0.1)
    # Feed shared stats where "b" has much lower success than "a".
    shared = {
        "a": {"n_samples": 10.0, "n_successes": 9.0, "success_rate": 0.9},
        "b": {"n_samples": 10.0, "n_successes": 1.0, "success_rate": 0.1},
    }
    picks = [sel.select(["a", "b"], context={"main_stats": shared}) for _ in range(5)]
    assert picks == ["b"] * 5


def test_make_component_selector_factory():
    assert isinstance(make_component_selector("ucb"), UCBComponentSelector)
    assert isinstance(make_component_selector("round_robin"), RoundRobinComponentSelector)
    assert isinstance(make_component_selector("stagnation"), StagnationComponentSelector)
    instance = UCBComponentSelector()
    assert make_component_selector(instance) is instance


def test_stats_shape():
    sel = UCBComponentSelector()
    sel.update("a", accepted=True)
    sel.update("a", accepted=False)
    sel.update("b", accepted=True)
    stats = sel.stats()
    assert stats["a"]["n_samples"] == 2
    assert stats["a"]["n_successes"] == 1
    assert stats["a"]["success_rate"] == 0.5
    assert stats["b"]["n_samples"] == 1
    assert stats["b"]["success_rate"] == 1.0
