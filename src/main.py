import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# Загрузка датасетов
df1 = pd.read_csv("C:/Users/alazi/lab_project/data/raw/car_data.csv")
df2 = pd.read_csv("C:/Users/alazi/lab_project/data/raw/Car_details_v3.csv")
df3 = pd.read_csv("C:/Users/alazi/lab_project/data/raw/CAR_DETAILS_FROM_CAR_DEKHO.csv")

# Колонки первого датасета
print("\nColumns df1:")
print(df1.columns)

# Колонки второго датасета
print("\nColumns df2:")
print(df2.columns)

# Колонки третьего датасета
print("\nColumns df3:")
print(df3.columns)

# Переименование колонок df1
df1 = df1.rename(columns={
    "Car_Name": "name",
    "Year": "year",
    "Selling_Price": "selling_price",
    "Kms_Driven": "km_driven",
    "Fuel_Type": "fuel",
    "Seller_Type": "seller_type",
    "Transmission": "transmission",
    "Owner": "owner"
})

# Выбор одинаковых колонок
common_columns = [
    "name",
    "year",
    "selling_price",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]
df1 = df1[common_columns]
df2 = df2[common_columns]
df3 = df3[common_columns]

# Объединение датасетов
merged_df = pd.concat([df1, df2, df3], ignore_index=True)

# Сохранение
merged_df.to_csv(
    "data/processed/merged_cars.csv",
    index=False
)
# Информация
print("\nДатасеты успешно объединены!")
print("\nРазмер датасета:")
print(merged_df.shape)

print("\nИнформация:")
print(merged_df.info())

print("\nПервые 5 строк:")
print(merged_df.head())

print("\nПропущенные значения:")
print(merged_df.isnull().sum())

# Создание папки для графиков
os.makedirs("reports/figures", exist_ok=True)

# Гистограмма цен
plt.figure(figsize=(10, 6))
sns.histplot(merged_df["selling_price"], bins=30)

plt.title("Distribution of Selling Price")
plt.xlabel("Selling Price")
plt.ylabel("Count")

plt.savefig("reports/figures/selling_price_distribution.png")

# Boxplot цены
plt.figure(figsize=(10, 6))
sns.boxplot(x=merged_df["selling_price"])

plt.title("Selling Price Boxplot")

plt.savefig("reports/figures/selling_price_boxplot.png")

# Количество машин по типу топлива
plt.figure(figsize=(8, 5))
sns.countplot(x=merged_df["fuel"])

plt.title("Fuel Type Distribution")

plt.savefig("reports/figures/fuel_distribution.png")

# Корреляция числовых признаков
numeric_df = merged_df.select_dtypes(include=["int64", "float64"])

plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")

plt.title("Correlation Matrix")

plt.savefig("reports/figures/correlation_matrix.png")

print("\nEDA графики сохранены!")


# Кодирование категориальных признаков

encoder = LabelEncoder()

categorical_columns = [
    "name",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]

for col in categorical_columns:
    merged_df[col] = merged_df[col].astype(str)
    merged_df[col] = encoder.fit_transform(merged_df[col])


# Разделение признаков

X = merged_df.drop("selling_price", axis=1)
y = merged_df["selling_price"]

# Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Linear Regression

model = LinearRegression()

model.fit(X_train, y_train)

# Предсказания

y_pred = model.predict(X_test)

# Метрики

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

print("\nLinear Regression Metrics:")

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")

# Decision Tree


tree_model = DecisionTreeRegressor(
    random_state=42,
    max_depth=10
)

tree_model.fit(X_train, y_train)

# Предсказания
tree_pred = tree_model.predict(X_test)

# Метрики
tree_mae = mean_absolute_error(y_test, tree_pred)

tree_rmse = np.sqrt(mean_squared_error(y_test, tree_pred))

tree_r2 = r2_score(y_test, tree_pred)

print("\nDecision Tree Metrics:")

print(f"MAE: {tree_mae:.4f}")
print(f"RMSE: {tree_rmse:.4f}")
print(f"R2 Score: {tree_r2:.4f}")


# CatBoost


cat_model = CatBoostRegressor(
    iterations=200,
    learning_rate=0.1,
    depth=6,
    verbose=0
)

cat_model.fit(X_train, y_train)

# Предсказания
cat_pred = cat_model.predict(X_test)

# Метрики
cat_mae = mean_absolute_error(y_test, cat_pred)

cat_rmse = np.sqrt(mean_squared_error(y_test, cat_pred))

cat_r2 = r2_score(y_test, cat_pred)

print("\nCatBoost Metrics:")

print(f"MAE: {cat_mae:.4f}")
print(f"RMSE: {cat_rmse:.4f}")
print(f"R2 Score: {cat_r2:.4f}")


# XGBoost

xgb_model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

xgb_model.fit(X_train, y_train)

# Предсказания
xgb_pred = xgb_model.predict(X_test)

# Метрики
xgb_mae = mean_absolute_error(y_test, xgb_pred)

xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))

xgb_r2 = r2_score(y_test, xgb_pred)

print("\nXGBoost Metrics:")

print(f"MAE: {xgb_mae:.4f}")
print(f"RMSE: {xgb_rmse:.4f}")
print(f"R2 Score: {xgb_r2:.4f}")

# Scaling данных

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# MLP Neural Network

mlp_model = MLPRegressor(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)

# Обучение
mlp_model.fit(X_train_scaled, y_train)

# Предсказания
mlp_pred = mlp_model.predict(X_test_scaled)

# Метрики
mlp_mae = mean_absolute_error(y_test, mlp_pred)

mlp_rmse = np.sqrt(mean_squared_error(y_test, mlp_pred))

mlp_r2 = r2_score(y_test, mlp_pred)

print("\nMLP Metrics:")

print(f"MAE: {mlp_mae:.4f}")
print(f"RMSE: {mlp_rmse:.4f}")
print(f"R2 Score: {mlp_r2:.4f}")

# Итоговая таблица метрик

metrics_df = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Decision Tree",
        "CatBoost",
        "XGBoost",
        "MLP"
    ],
    "MAE": [
        mae,
        tree_mae,
        cat_mae,
        xgb_mae,
        mlp_mae
    ],
    "RMSE": [
        rmse,
        tree_rmse,
        cat_rmse,
        xgb_rmse,
        mlp_rmse
    ],
    "R2 Score": [
        r2,
        tree_r2,
        cat_r2,
        xgb_r2,
        mlp_r2
    ]
})

# Сохранение таблицы
metrics_df.to_csv(
    "reports/metrics.csv",
    index=False
)

# Вывод таблицы
print("\nИтоговая таблица метрик:")
print(metrics_df)

# График сравнения моделей по R2

plt.figure(figsize=(10, 6))
sns.barplot(
    data=metrics_df,
    x="Model",
    y="R2 Score"
)

plt.title("Model Comparison by R2 Score")
plt.xlabel("Model")
plt.ylabel("R2 Score")
plt.xticks(rotation=30)

plt.savefig("reports/figures/model_comparison_r2.png")
plt.close()

print("\nГрафик сравнения моделей сохранён!")

# Feature Importance XGBoost

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgb_model.feature_importances_
})

# Сортировка
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

# Сохранение
importance_df.to_csv(
    "reports/feature_importance_xgboost.csv",
    index=False
)

# График
plt.figure(figsize=(10, 6))

sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature"
)

plt.title("XGBoost Feature Importance")

plt.savefig(
    "reports/figures/xgboost_feature_importance.png"
)

plt.close()

print("\nFeature Importance сохранён!")

# Feature Importance CatBoost

cat_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": cat_model.get_feature_importance()
})

cat_importance_df = cat_importance_df.sort_values(
    by="Importance",
    ascending=False
)

cat_importance_df.to_csv(
    "reports/feature_importance_catboost.csv",
    index=False
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=cat_importance_df,
    x="Importance",
    y="Feature"
)

plt.title("CatBoost Feature Importance")

plt.savefig(
    "reports/figures/catboost_feature_importance.png"
)

plt.close()

print("\nCatBoost Feature Importance сохранён!")


# =========================
# Сохранение моделей
# =========================

joblib.dump(model, "models/linear_regression.pkl")

joblib.dump(tree_model, "models/decision_tree.pkl")

joblib.dump(cat_model, "models/catboost.pkl")

joblib.dump(xgb_model, "models/xgboost.pkl")

joblib.dump(mlp_model, "models/mlp.pkl")

print("\nВсе модели сохранены!")
