from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


def build_pipeline(preprocessor) -> Pipeline:
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(n_estimators=100, random_state=42)),
    ])
    return pipeline


def train(pipeline: Pipeline, X_train, y_train) -> Pipeline:
    pipeline.fit(X_train, y_train)
    return pipeline