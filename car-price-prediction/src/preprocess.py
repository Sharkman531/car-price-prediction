import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Extract brand name from CarName
    df["brand"] = df["CarName"].str.split().str[0].str.lower()
    df = df.drop(columns=["car_ID", "CarName"])
    return df


def get_features_and_target(df: pd.DataFrame):
    X = df.drop(columns=["price"])
    y = df["price"]
    return X, y


def get_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    numerical_cols = X.select_dtypes(include="number").columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])
    return preprocessor


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)