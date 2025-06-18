import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import add_distance_feature
from src.models.train_linear import train_lr
from src.models.train_random_forest import train_rf
from src.models.train_xgboost import train_xgb
from src.evaluate import evaluate_model
from src.model_selector import select_best_model
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib




def main():
    df = load_and_clean_data("data/deliverytime.csv")
    df = add_distance_feature(df)

    X = pd.get_dummies(df[['Delivery_person_Age', 'Delivery_person_Ratings', 'distance_km', 'Type_of_order', 'Type_of_vehicle']], drop_first=True)
    y = df['Time_taken(min)']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = [
        ("Linear", train_lr(X_train, y_train)),
        ("Random Forest", train_rf(X_train, y_train)),
        ("XGBoost", train_xgb(X_train, y_train))
    ]

    results = []
    for name, model in models:
        scores = evaluate_model(model, X_test, y_test)
        results.append((name, model, scores))
        print(f"{name} MAE: {scores['mae']:.2f} | RMSE: {scores['rmse']:.2f} | R²: {scores['r2']:.2f}")

    best_name, best_model, best_scores = select_best_model(results)
    print(f"\n✅ Best model: {best_name} with MAE {best_scores['mae']:.2f}")
    joblib.dump(best_model, "models/best_model.pkl")

if __name__ == "__main__":
    main()
