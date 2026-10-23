"""Deprecated evaluator for the leaky NGO prepared split.

The audited grouped test report is created by train_audited_model.py. Running
an evaluator against the NGO prepared split would reuse near-duplicate groups
and overwrite the current report with a non-independent score.
"""

if __name__ == "__main__":
    print("Legacy evaluation is disabled because the NGO prepared split has group leakage risk.")
    print("Use reports/evaluation_v2/ and the grouped test split from train_audited_model.py.")
