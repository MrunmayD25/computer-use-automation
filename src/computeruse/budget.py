"""Limits that stop a run, counted against the operator's declared budgets.

The caller receives a distinct exhaustion result when a run reaches a limit,
so it cannot mistake a stopped run for a completed one.
"""

import contextlib
from collections.abc import Callable, Iterator
from enum import StrEnum

from computeruse.profile import Budgets

type Clock = Callable[[], float]
"""Monotonic seconds. Injected so wall-clock limits are testable without sleeping."""


class Exhaustion(StrEnum):
    """Which declared limit ended the run."""

    MAX_STEPS = "max_steps"
    MAX_WALL_CLOCK = "max_wall_clock_s"
    MAX_NAVIGATIONS = "max_navigations"


class Budget:
    """Track one run's use of the limits in a profile's ``budgets`` section."""

    __slots__ = (
        "_clock",
        "_depth",
        "_held_since",
        "_limits",
        "_navigations",
        "_paused",
        "_started",
        "_steps",
    )

    def __init__(self, limits: Budgets, clock: Clock) -> None:
        self._limits = limits
        self._clock = clock
        self._started = clock()
        self._steps = 0
        self._navigations = 0
        self._paused = 0.0
        self._depth = 0
        self._held_since = 0.0

    @property
    def steps(self) -> int:
        """How many decisions the run has taken so far."""
        return self._steps

    def exhausted(
        self, *, starting_step: bool = True, navigating: bool = False
    ) -> Exhaustion | None:
        """Return the limit that has been reached, or None while headroom remains.

        Parameters
        ----------
        starting_step
            Check the step limit before reserving a decision. Set False while
            processing that decision, so its action can use the final step.
        navigating
            Also check the navigation limit. Check on demand so a run that has
            used all its navigations can still read the current screen.
        """
        if starting_step and self._steps >= self._limits.max_steps:
            return Exhaustion.MAX_STEPS
        if self.remaining_seconds() <= 0:
            return Exhaustion.MAX_WALL_CLOCK
        if navigating and self._navigations >= self._limits.max_navigations:
            return Exhaustion.MAX_NAVIGATIONS
        return None

    def remaining_seconds(self) -> float:
        """Return the time available for the next blocking operation."""
        return max(0.0, self._limits.max_wall_clock_s - self.active_seconds())

    def active_seconds(self) -> float:
        """Return elapsed time with human waiting removed.

        A run's wall-clock budget limits how long the automation may work, not
        how long an operator may take to answer. Counting the wait would make
        the budget punish the very handoff the policy asked for, and would let
        a slow operator end a run the automation could have finished.

        A pause still in progress is removed too, so the remaining time an
        operator is shown does not fall while they hold the run.
        """
        return self._clock() - self._started - self.paused_seconds()

    def paused_seconds(self) -> float:
        """Return the time spent waiting for a person, including a pause now."""
        if self._depth == 0:
            return self._paused
        return self._paused + max(0.0, self._clock() - self._held_since)

    def pause(self) -> None:
        """Stop the execution clock, counting nested pauses once.

        Two things can hold the clock at the same moment: the run's control,
        while a person holds the run, and an operator channel that answers
        synchronously. Only the outermost pause measures time, so a wait
        inside a wait is not subtracted twice.
        """
        if self._depth == 0:
            self._held_since = self._clock()
        self._depth += 1

    def unpause(self) -> None:
        """End one pause, restarting the clock when the outermost one ends."""
        if self._depth == 0:
            return
        self._depth -= 1
        if self._depth == 0:
            self._paused += max(0.0, self._clock() - self._held_since)

    @contextlib.contextmanager
    def waiting_for_a_human(self) -> Iterator[None]:
        """Stop the execution clock while a person holds the session."""
        self.pause()
        try:
            yield
        finally:
            self.unpause()

    def charge_step(self) -> None:
        """Count one decision, whatever became of it."""
        self._steps += 1

    def refund_step(self) -> None:
        """Take back a step whose decision was discarded before anything read it.

        An operator who stops a run while the model is deciding has not had
        that decision taken, so the run's step count is left where it was.
        """
        self._steps = max(0, self._steps - 1)

    def charge_navigation(self) -> None:
        """Count one navigation that actually reached the surface."""
        self._navigations += 1
