# Data Cleaning & Preprocessing:-

## 📌 Overview

This project focuses on converting the **Titanic dataset** from its raw form into a **machine‑learning‑ready format** by performing systematic **data cleaning and preprocessing techniques** in a production‑style Python pipeline.

The goal of this task is to demonstrate core real‑world preprocessing skills including:

- Handling missing values
- Encoding categorical variables
- Scaling numerical features
- Visualizing outliers and correlations
- Exporting a fully cleaned dataset

---

## 🧠 Project Workflow



1. Load and inspect dataset
2. Identify and handle missing values (median/mode/drop)
3. Encode categorical features using One‑Hot Encoding
4. Standardize numerical features with Standard Scaler
5. Generate visualizations for data understanding
6. Export cleaned dataset for ML model usage

---

## 📂 Folder Structure

```
AIML_INTERNSHIP_TASK_1_ELEVATE_LABS
│
├── Dataset
│     └── Titanic-Dataset.csv
│
├── Output
│     ├── cleaned_titanic_dataset.csv
│     ├── boxplots_numerical_features.png
│     ├── correlation_heatmap.png
│     └── survival_rate_by_sex.png
│
└── titanic_preprocessing.py
```

---

## 🔍 Visual Results (Screenshots)

| Visualization       | Description                                              |
| ------------------- | -------------------------------------------------------- |
| Box plots           | Detects outliers in numerical features                   |
| Heat map            | Shows correlation among all numerical & encoded features |
| Survival Rate Chart | Reveals gender‑based survival differences                |

All screenshots are inside the **Output** folder.

---

## 🛠 Tech Stack

| Component | Technology                                       |
| --------- | ------------------------------------------------ |
| Language  | Python                                           |
| Libraries | Pandas, NumPy, Matplotlib, Seaborn, Scikit‑Learn |
| IDE       | Visual Studio Code                               |

---

## 📦 Requirements

Install dependencies before running the script:

```
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## ▶ Running the Script

Make sure the dataset is inside the **Dataset** folder. Then run:

```
python titanic_preprocessing.py
```

After execution, cleaned data and visual outputs will appear inside the **Output** folder.

---

## 📌 Outcome

The final cleaned dataset contains: ✔ zero missing values ✔ all features numeric (no strings) ✔ scaled numerical data ✔ suitable for ML algorithms like Logistic Regression, SVM, Random Forest, etc.

---

## 👤 Author

**Name:** Ullas B R, **Role:** AIML Internship Participant — Elevate Labs, **Task 1:** Data Cleaning & Preprocessing

---

## ⭐ Final Note

This project demonstrates end‑to‑end preprocessing in a **real deployment‑style structure**, ensuring reproducibility and engineering‑level data preparation. Feel free to explore, modify, and build ML models on top of this cleaned dataset.

