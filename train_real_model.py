"""Deprecated training entry point.

The NGO prepared split contains repeated photo groups across train/validation/test
and must not be used for retraining. Use train_audited_model.py, which reads only
accepted records from the primary `to review real dataset` source and creates
leakage-resistant group splits.
"""

if __name__ == "__main__":
    print("Legacy training is disabled because the prepared NGO split has group leakage risk.")
    print("Run audit_real_dataset.py followed by train_audited_model.py instead.")
