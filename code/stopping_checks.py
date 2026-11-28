"""Shared checkpoint schedule for the first certified stopping time."""


def is_checkpoint(iteration, period, last):
    return iteration == 1 or iteration % period == 0 or iteration == last


def checkpoint_count(iterations, period=10):
    """Count distinct checks at 1, each period, and the reported endpoint."""
    if iterations < 1 or period < 1:
        raise ValueError("iterations and period must be positive")
    count = iterations // period + int(period != 1)
    return count + int(iterations > 1 and iterations % period != 0)
