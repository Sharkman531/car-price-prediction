import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score


def print_metrics(y_test, y_pred):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R²:   {r2:.4f}")


def plot_actual_vs_predicted(y_test, y_pred, output_path="plots/actual_vs_predicted.html"):
    df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})

    fig = px.scatter(
        df, x="Actual", y="Predicted",
        title="Actual vs Predicted Car Price",
        labels={"Actual": "Actual Price (USD)", "Predicted": "Predicted Price (USD)"},
        opacity=0.7,
        color_discrete_sequence=["#636EFA"],
    )
    # Perfect prediction reference line
    min_val = min(df["Actual"].min(), df["Predicted"].min())
    max_val = max(df["Actual"].max(), df["Predicted"].max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines", name="Perfect Prediction",
        line=dict(color="red", dash="dash")
    ))
    fig.write_html(output_path)
    print(f"Plot saved: {output_path}")


def plot_feature_importance(pipeline, X_train, output_path="plots/feature_importance.html"):
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    # Recover feature names after encoding
    num_features = preprocessor.transformers_[0][2]
    cat_features = preprocessor.named_transformers_["cat"].get_feature_names_out(
        preprocessor.transformers_[1][2]
    ).tolist()
    all_features = list(num_features) + cat_features

    importances = pd.Series(model.feature_importances_, index=all_features)
    importances = importances.sort_values(ascending=False).head(15)

    fig = px.bar(
        importances.reset_index(),
        x="index", y=0,
        title="Top 15 Feature Importances",
        labels={"index": "Feature", "0": "Importance"},
        color_discrete_sequence=["#636EFA"],
    )
    fig.write_html(output_path)
    print(f"Plot saved: {output_path}")


def plot_price_by_brand(df, output_path="plots/price_by_brand.html"):
    avg_price = (
        df.groupby("brand")["price"]
        .median()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig = px.bar(
        avg_price, x="brand", y="price",
        title="Median Car Price by Brand",
        labels={"brand": "Brand", "price": "Median Price (USD)"},
        color="price",
        color_continuous_scale="Blues",
    )
    fig.write_html(output_path)
    print(f"Plot saved: {output_path}")