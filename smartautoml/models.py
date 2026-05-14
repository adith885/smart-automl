from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR

def get_models(task):

    if task == "classification":
        return {
            "log_reg": LogisticRegression(max_iter=1000),
            "rf": RandomForestClassifier(),
            "svm": SVC()
        }

    return {
        "linear": LinearRegression(),
        "rf": RandomForestRegressor(),
        "svr": SVR()
    }


def get_param_grids(task):

    if task == "classification":
        return {
            "log_reg": {"model__C": [0.1, 1, 10]},
            "rf": {"model__n_estimators": [50, 100]},
            "svm": {"model__C": [0.1, 1, 10]}
        }

    return {
        "linear": {},
        "rf": {"model__n_estimators": [50, 100]},
        "svr": {"model__C": [0.1, 1, 10]}
    }