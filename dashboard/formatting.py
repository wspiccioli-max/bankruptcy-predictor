import pandas as pd


PROBABILITY_DISCLAIMER = (
    "Model probabilities are estimated filing-risk similarity scores, not certainty "
    "or investment advice."
)


def format_probability(value, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    if value < 0.01:
        return "<1%"
    if value > 0.99:
        return ">99%"
    return f"{value:.{digits}%}"


def format_delta(value, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:+.{digits}%}"
