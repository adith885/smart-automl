import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR


class SmartAutoML:

    def __init__(self, cv=3):
        self.cv = cv
        self.best_model = None
        self.task = None
        self.best_score = None

    # -------------------------
    # TASK DETECTION
    # -------------------------
    def detect_task(self, y):

        y = pd.Series(y).dropna().values
        unique = len(np.unique(y))

        if y.dtype == object or unique <= 10:
            return "classification"

        return "regression"

    # -------------------------
    # SAFE CV
    # -------------------------
    def get_safe_cv(self, y, task):

        n = len(y)

        if task == "classification":
            _, counts = np.unique(y, return_counts=True)
            min_class = min(counts)

            cv = min(self.cv, min_class)

            if cv < 2:
                cv = 2

            return StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        else:
            cv = min(self.cv, n)

            if cv < 2:
                cv = 2

            return KFold(n_splits=cv, shuffle=True, random_state=42)

    # -------------------------
    # MODELS
    # -------------------------
    def get_models(self, task):

        if task == "classification":
            return {
                "log_reg": LogisticRegression(max_iter=1000),
                "rf": RandomForestClassifier(),
                "svm": SVC()
            }

        else:
            return {
                "lin_reg": LinearRegression(),
                "rf": RandomForestRegressor(),
                "svr": SVR()
            }

    # -------------------------
    # FIT (WITH ERROR HANDLING)
    # -------------------------
    def fit(self, df, target_column):

        # ❌ 1. TARGET COLUMN CHECK
        if target_column not in df.columns:
            print(f"❌ Error: Target column '{target_column}' not found.")
            print(f"Available columns: {list(df.columns)}")
            return self

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # ❌ 2. EMPTY TARGET CHECK
        if y.isna().all():
            print(f"❌ Error: Target column '{target_column}' has only NaN values.")
            return self

        # remove NaN target rows
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]

        # detect task
        self.task = self.detect_task(y)
        print(f"Detected task: {self.task}")

        # train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # auto column detection
        numeric_features = X.select_dtypes(include=[np.number]).columns
        categorical_features = X.select_dtypes(exclude=[np.number]).columns

        # preprocessing
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features)
            ]
        )

        models = self.get_models(self.task)

        best_score = -np.inf
        best_model = None

        # -------------------------
        # TRAIN MODELS SAFELY
        # -------------------------
        for name, model in models.items():

            print(f"Training model: {name}")

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])

            # ❌ 3. SAFE CV HANDLING
            try:
                cv_strategy = self.get_safe_cv(y_train, self.task)
            except Exception as e:
                print("❌ CV Error:", str(e))
                print("⚠️ Using fallback CV=2")
                cv_strategy = KFold(n_splits=2, shuffle=True, random_state=42)

            grid = GridSearchCV(
                pipeline,
                param_grid={},
                cv=cv_strategy,
                n_jobs=-1,
                error_score="raise"
            )

            # ❌ 4. SAFE TRAINING WRAPPER
            try:
                grid.fit(X_train, y_train)
            except Exception as e:
                print(f"❌ Training failed for {name}: {str(e)}")
                continue

            score = grid.score(X_test, y_test)

            print(f"{name} score: {score}")

            if score > best_score:
                best_score = score
                best_model = grid.best_estimator_

        self.best_model = best_model
        self.best_score = best_score

        print(f"\nBEST MODEL SCORE: {self.best_score}")

        return self

    # -------------------------
    # PREDICT
    # -------------------------
    def predict(self, X):
        return self.best_model.predict(X)

    # -------------------------
    # SAVE MODEL
    # -------------------------
    def save(self, path="model.pkl"):
        import joblib
        joblib.dump(self.best_model, path)
        print(f"Model saved at {path}")