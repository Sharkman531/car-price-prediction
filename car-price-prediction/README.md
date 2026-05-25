# Car Price Prediction · ML Pipeline

Machine learning pipeline predicting car prices from the Kaggle Car Price Assignment dataset,
built with modular Python scripts using scikit-learn, Pandas, and Plotly.

## Tools
Python · scikit-learn · Pandas · NumPy · Plotly

## Dataset
[Car Price Assignment](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction) from Kaggle.
Contains 205 cars across 26 features including brand, fuel type, engine specs, body style, and price (USD).

## Project Structure
```
car-price-prediction/
├── data/
│   └── CarPrice_Assignment.csv
├── src/
│   ├── preprocess.py   # Data loading, brand extraction, ColumnTransformer
│   ├── train.py        # Pipeline assembly and model training
│   └── evaluate.py     # Metrics and Plotly visualizations
├── plots/
│   ├── actual_vs_predicted.html
│   ├── feature_importance.html
│   └── price_by_brand.html
├── main.py
├── requirements.txt
└── README.md
```

## Pipeline
- Brand name extracted from `CarName` field as an additional feature
- `ColumnTransformer`: `StandardScaler` for numerical features, `OneHotEncoder` for categorical features
- Model: `RandomForestRegressor` (100 estimators)
- Train/test split: 80/20

## Model Performance
| Metric | Value |
|--------|-------|
| RMSE   | $1,851.81 |
| R²     | 0.9566 |

## Visualizations
All plots are interactive HTML files generated with Plotly:
- **Actual vs Predicted** — scatter plot with perfect prediction reference line
- **Feature Importances** — top 15 most influential features in the model
- **Median Price by Brand** — bar chart comparing median price across all brands

## Key Findings
- The model explains 95.7% of price variance with a mean error of ~$1,852
- Engine size, curb weight, and horsepower are the strongest price predictors
- Brand alone is a significant factor: luxury brands show median prices 3–4× higher than economy brands
- Fuel type, drive wheel, and body style have comparatively low predictive importance