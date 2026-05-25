# 🚗 ML-проект: Предсказание стоимости автомобилей

## 📌 Описание проекта

Данный проект посвящён предсказанию стоимости автомобилей с использованием методов машинного обучения.

В проекте реализованы:

- объединение нескольких датасетов;
- анализ данных (EDA);
- preprocessing и feature engineering;
- обучение моделей машинного обучения;
- сравнение метрик;
- анализ важности признаков;
- сохранение моделей;
- DVC pipeline.

Цель проекта — определить наиболее эффективную модель для задачи предсказания цены автомобиля.

---

# 📂 Используемые датасеты

В проекте используются 3 датасета:

- `car_data.csv`
- `Car_details_v3.csv`
- `CAR_DETAILS_FROM_CAR_DEKHO.csv`

После обработки датасеты были объединены в единый набор данных:

```text
data/processed/merged_cars.csv
```

---

# ⚙️ Предобработка данных

В процессе preprocessing были выполнены:

- унификация названий колонок;
- объединение датасетов;
- анализ пропущенных значений;
- кодирование категориальных признаков (`LabelEncoder`);
- нормализация признаков (`StandardScaler`) для MLP.

---

# 📊 Разведочный анализ данных (EDA)

В рамках EDA были построены:

- распределение цен автомобилей;
- boxplot стоимости;
- анализ типов топлива;
- корреляционная матрица;
- график сравнения моделей.

Все графики сохранены в:

```text
reports/figures/
```

---

# 🤖 Используемые модели

В проекте реализованы следующие модели:

1. Linear Regression
2. Decision Tree Regressor
3. CatBoost Regressor
4. XGBoost Regressor
5. MLP Neural Network

---

# 📈 Метрики моделей

| Модель | R² Score |
|---|---|
| Linear Regression | 0.4371 |
| Decision Tree | 0.8096 |
| CatBoost | 0.8578 |
| XGBoost | 0.8918 |
| MLP | 0.6319 |

Лучшая модель: **XGBoost**

---

# 📉 Анализ важности признаков

Feature Importance был построен для:

- XGBoost
- CatBoost

Файлы результатов:

```text
reports/feature_importance_xgboost.csv
reports/feature_importance_catboost.csv
```

---

# 💾 Сохранённые модели

Все обученные модели сохранены в папке:

```text
models/
```

Список моделей:

- `linear_regression.pkl`
- `decision_tree.pkl`
- `catboost.pkl`
- `xgboost.pkl`
- `mlp.pkl`

---

# 📁 Структура проекта

```text
lab_ai_project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── reports/
│   ├── figures/
│   └── metrics.csv
│
├── src/
│   └── main.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# ▶️ Установка проекта

Создание виртуального окружения:

```bash
python -m venv .venv
```

Активация окружения:

```bash
.venv\Scripts\activate
```

Установка библиотек:

```bash
pip install -r requirements.txt
```

---

# ▶️ Запуск проекта

```bash
python src/main.py
```

---

# 📌 Результаты

В ходе работы было установлено, что ансамблевые методы градиентного бустинга значительно превосходят линейные модели для задачи предсказания стоимости автомобилей.

Наилучший результат показала модель:

```text
XGBoost — R² = 0.8918
```

---

# 🔄 DVC Pipeline

В проекте будет реализован DVC pipeline для отслеживания экспериментов и воспроизводимости ML workflow.

---

# 🛠 Используемые технологии

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- CatBoost
- XGBoost
- TensorFlow
- DVC

---

# 👨‍💻 Автор

Лабораторная работа по дисциплине  
«Методы искусственного интеллекта»