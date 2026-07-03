"""
markov_progression.py
Replaces the old "just count percentages per timepoint" logic in Feature 4
(Accelerated Timelines) with a real discrete-time Markov chain model of
glycemic control state over time.

States (based on ADA HbA1c targets):
    "Controlled"    : HbA1c < 7.0%
    "Borderline"    : 7.0% <= HbA1c < 8.0%
    "Uncontrolled"  : HbA1c >= 8.0%

The transition matrix is learned empirically from the synthetic cohort by
looking at how patients move between states across consecutive follow-up
visits (baseline -> 3m -> 6m -> 12m -> 24m -> 36m). Once learned, the chain
can be simulated forward beyond the observed data (e.g. to 48m, 60m) to
project long-term population outcomes -- something the old percentage-count
approach could never do, since it only reads existing columns.
"""

import numpy as np
import pandas as pd

STATES = ["Controlled", "Borderline", "Uncontrolled"]
STATE_INDEX = {s: i for i, s in enumerate(STATES)}

TIMEPOINT_COLUMNS = [
    "hba1c_baseline", "hba1c_3_months", "hba1c_6_months",
    "hba1c_12_months", "hba1c_24_months", "hba1c_36_months",
]
TIMEPOINT_MONTHS = [0, 3, 6, 12, 24, 36]


def classify_state(hba1c: float) -> str:
    if hba1c < 7.0:
        return "Controlled"
    elif hba1c < 8.0:
        return "Borderline"
    else:
        return "Uncontrolled"


def build_transition_matrix(df: pd.DataFrame, group: str = None) -> np.ndarray:
    """
    Learn an empirical transition matrix from the cohort.
    If `group` is given ("Treatment" or "Control"), the matrix is learned
    only from patients in that arm -- so treatment and control project
    differently, which is clinically meaningful.
    """
    data = df if group is None else df[df["treatment_group"] == group]

    counts = np.zeros((len(STATES), len(STATES)))

    for _, row in data.iterrows():
        states_sequence = []
        for col in TIMEPOINT_COLUMNS:
            val = row.get(col)
            if pd.isna(val):
                states_sequence.append(None)
            else:
                states_sequence.append(classify_state(float(val)))

        for i in range(len(states_sequence) - 1):
            s_from, s_to = states_sequence[i], states_sequence[i + 1]
            if s_from is None or s_to is None:
                continue
            counts[STATE_INDEX[s_from], STATE_INDEX[s_to]] += 1

    # Normalize rows to probabilities; if a state was never observed, assume it stays put
    matrix = np.zeros_like(counts)
    for i in range(len(STATES)):
        row_sum = counts[i].sum()
        if row_sum > 0:
            matrix[i] = counts[i] / row_sum
        else:
            matrix[i, i] = 1.0
    return matrix


def initial_distribution(df: pd.DataFrame, group: str = None) -> np.ndarray:
    data = df if group is None else df[df["treatment_group"] == group]
    states = data["hba1c_baseline"].dropna().apply(classify_state)
    dist = np.array([
        (states == s).sum() for s in STATES
    ], dtype=float)
    if dist.sum() == 0:
        return np.array([0.0, 0.0, 1.0])
    return dist / dist.sum()


def simulate_forward(transition_matrix: np.ndarray, start_dist: np.ndarray, n_steps: int):
    """
    Simulate the population state distribution forward `n_steps` transitions.
    Each step corresponds to one observed interval in the data (roughly the
    spacing between the empirical timepoints); for projecting beyond month 36
    we keep applying the same matrix, which is the standard steady-state-
    projection use of a Markov chain.
    """
    distributions = [start_dist]
    current = start_dist
    for _ in range(n_steps):
        current = current @ transition_matrix
        distributions.append(current)
    return distributions


def project_timeline(df: pd.DataFrame, extra_steps: int = 2):
    """
    Builds a month-by-month projected distribution for Treatment and Control
    groups, including projection beyond the last observed timepoint (36m).

    Returns a dict ready to be sent to the frontend as JSON.
    """
    months = TIMEPOINT_MONTHS.copy()
    # Extend the timeline by repeating the last interval spacing (12 months)
    for _ in range(extra_steps):
        months.append(months[-1] + 12)

    output = {}
    for group in ["Treatment", "Control"]:
        tm = build_transition_matrix(df, group=group)
        start = initial_distribution(df, group=group)
        dists = simulate_forward(tm, start, n_steps=len(TIMEPOINT_COLUMNS) - 1 + extra_steps)

        output[group] = {
            "transition_matrix": {
                STATES[i]: {STATES[j]: round(float(tm[i, j]), 3) for j in range(len(STATES))}
                for i in range(len(STATES))
            },
            "timeline": [
                {
                    "month": months[t],
                    "Controlled": round(float(dists[t][STATE_INDEX["Controlled"]]) * 100, 1),
                    "Borderline": round(float(dists[t][STATE_INDEX["Borderline"]]) * 100, 1),
                    "Uncontrolled": round(float(dists[t][STATE_INDEX["Uncontrolled"]]) * 100, 1),
                }
                for t in range(len(months))
            ],
        }
    return output
