from src.preprocess import load_data, get_features_and_target, get_preprocessor, split_data
from src.train import build_pipeline, train
from src.evaluate import print_metrics, plot_actual_vs_predicted, plot_feature_importance, plot_price_by_brand

DATA_PATH = "data/CarPrice_Assignment.csv"

def main():
    # 1. Load and prepare data
    print("Loading data...")
    df = load_data(DATA_PATH)
    X, y = get_features_and_target(df)

    # 2. Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 3. Build and train pipeline
    print("Training pipeline...")
    preprocessor = get_preprocessor(X_train)
    pipeline = build_pipeline(preprocessor)
    pipeline = train(pipeline, X_train, y_train)

    # 4. Evaluate
    print("\n--- Model Performance ---")
    y_pred = pipeline.predict(X_test)
    print_metrics(y_test, y_pred)

    # 5. Generate plots
    print("\nGenerating plots...")
    plot_actual_vs_predicted(y_test, y_pred)
    plot_feature_importance(pipeline, X_train)
    plot_price_by_brand(df)

    print("\nDone.")

if __name__ == "__main__":
    main()