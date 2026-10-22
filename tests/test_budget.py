import pytest
from hypothesis import given
from hypothesis import strategies as st

from rag_mcp.budget import Budget


class Clock:
    def __init__(self, t: float = 1_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def test_minute_window_refuses_then_recovers():
    clock = Clock()
    b = Budget(per_minute=2, clock=clock)
    assert [b.try_acquire() for _ in range(3)] == [None, None, "minute"]
    clock.t += 60
    assert b.try_acquire() is None


def test_daily_cap_and_reset_at_day_boundary():
    clock = Clock(86400 * 10)
    b = Budget(per_minute=100, per_day=2, clock=clock)
    assert [b.try_acquire() for _ in range(3)] == [None, None, "day"]
    clock.t = 86400 * 11
    assert b.try_acquire() is None


def test_refused_calls_do_not_consume_budget():
    clock = Clock()
    b = Budget(per_minute=1, per_day=2, clock=clock)
    assert b.try_acquire() is None
    assert b.try_acquire() == "minute"
    clock.t += 60
    assert b.try_acquire() is None  # the refused call above did not count towards the day


@pytest.mark.parametrize(("m", "d"), [(0, None), (1, 0)])
def test_rejects_non_positive_limits(m, d):
    with pytest.raises(ValueError):
        Budget(per_minute=m, per_day=d)


@given(
    per_minute=st.integers(1, 5),
    per_day=st.integers(1, 20),
    gaps=st.lists(st.floats(0, 90, allow_nan=False), min_size=1, max_size=80),
)
def test_property_limits_never_exceeded(per_minute, per_day, gaps):
    clock = Clock(0.0)
    b = Budget(per_minute=per_minute, per_day=per_day, clock=clock)
    granted: list[float] = []
    for gap in gaps:
        clock.t += gap
        if b.try_acquire() is None:
            granted.append(clock.t)
    for t in granted:
        assert sum(1 for g in granted if t <= g < t + 60) <= per_minute
    by_day: dict[int, int] = {}
    for t in granted:
        by_day[int(t // 86400)] = by_day.get(int(t // 86400), 0) + 1
    assert all(n <= per_day for n in by_day.values())
