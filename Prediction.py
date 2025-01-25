import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("laptop.csv")

plt.figure(figsize=(10, 6))
sns.histplot(data['Price'], bins=30, kde=True)
plt.title("Distribution of Laptop Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='TypeName', y='Price', data=data)
plt.title("Price Variation by Laptop Type")
plt.xticks(rotation=45)
plt.show()

data = data.drop(columns=["Unnamed: 0.1", "Unnamed: 0"])
data["Inches"] = pd.to_numeric(data["Inches"], errors="coerce")
data["Weight"] = pd.to_numeric(data["Weight"].str.replace("kg", ""), errors="coerce")
data = data.dropna()

X = data.drop(columns=["Price"])
y = data["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numerical_features = ["Inches", "Weight"]
categorical_features = ["Company", "TypeName", "ScreenResolution", "Cpu", "Ram", "Memory", "Gpu", "OpSys"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

from sklearn.model_selection import GridSearchCV

param_grid = {
    'regressor__n_estimators': [50, 100, 200],
    'regressor__max_depth': [None, 10, 20, 30],
    'regressor__min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best R² Score: {grid_search.best_score_}")

numerical_features_transformed = numerical_features
categorical_features_transformed = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist()
all_features = numerical_features_transformed + categorical_features_transformed

feature_importance = pd.DataFrame({
    'Feature': all_features,
    'Importance': pipeline.named_steps['regressor'].feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
plt.title('Top 10 Important Features')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

model_path = "laptop_price_model.pkl"
joblib.dump(pipeline, model_path)
print(f"Model saved to {model_path}")
