from imblearn.over_sampling import SMOTE

def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X_train, y_train)
