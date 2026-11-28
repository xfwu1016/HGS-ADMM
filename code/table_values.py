"""Numeric display shared by the empirical table generators."""
import math


def sig(value):
    """Three significant figures, retaining useful decimal zeros."""
    x = float(value)
    if not math.isfinite(x):
        return '--'
    if x == 0:
        return '0'
    exponent = math.floor(math.log10(abs(x)))
    if exponent < -3:
        mantissa = x / 10**exponent
        return rf'\({mantissa:.2f}\times10^{{{exponent}}}\)'
    decimals = max(0, 2 - exponent)
    rounded = round(x, 2 - exponent)
    return f'{rounded:,.{decimals}f}'


def iteration_mean(value):
    """Display the mean iteration count to the nearest whole iteration."""
    return f"{float(value):,.0f}"
