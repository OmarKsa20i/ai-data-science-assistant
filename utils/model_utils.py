"""Model training, evaluation, and persistence helpers."""

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor


def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train):
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def get_regression_metrics(y_true, predictions):
    return r2_score(y_true, predictions), mean_absolute_error(y_true, predictions)


def get_accuracy(y_true, predictions):
    return accuracy_score(y_true, predictions)


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)
