"""Fit preprocessing on the labeled training subset; select C on validation."""
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from src.data_loader import LABELS


def build_classifier(features, classifier, C, seed):
    preprocessing = (TfidfVectorizer(ngram_range=(1, 2), lowercase=True,
                                   min_df=1, sublinear_tf=True, max_features=30000)
                     if features == "tfidf" else StandardScaler())
    estimator = (LogisticRegression(C=C, max_iter=3000, class_weight="balanced",
                                    random_state=seed, solver="lbfgs")
                 if classifier == "lr" else
                 LinearSVC(C=C, max_iter=10000, class_weight="balanced",
                           random_state=seed, dual="auto"))
    return Pipeline([("preprocessing", preprocessing), ("classifier", estimator)])


def tune_classifier(X_train, y_train, X_val, y_val, features, classifier, grid, seed):
    best, best_score, trials = None, -1, []
    for C in grid:
        model = build_classifier(features, classifier, C, seed)
        model.fit(X_train, y_train)
        score = f1_score(y_val, model.predict(X_val), labels=LABELS,
                         average="macro", zero_division=0)
        trials.append({"C": C, "validation_macro_f1": float(score)})
        if score > best_score:
            best, best_score = model, score
    return best, trials
