# Titanic Survival Prediction using Machine Learning

## 1. Project Overview

This repository contains a data science project that builds a machine learning model to predict the survival probability of passengers on the Titanic. The entire workflow, from data exploration to an interactive prediction function, is documented in the accompanying Jupyter Notebook.

This project demonstrates some of the key skills I learned in the area of data analysis, feature engineering, and predictive modeling using Python and popular libraries like Pandas, Matplotlib, and Scikit-learn.

---

## 2. The Workflow

The Jupyter Notebook (`titanic_survival_prediction.ipynb`) is structured to follow a clear data science methodology:

1.  **Exploratory Data Analysis (EDA):** Investigating the dataset to understand its structure, identify missing values, and uncover initial insights.
2.  **Feature Engineering:** Creating new features (like `AgeGroup`) and transforming existing ones (like `Sex`) into a machine-readable format to improve model performance.
3.  **Model Training:** Building a **Logistic Regression** model, a suitable algorithm for binary classification, to learn patterns from the data.
4.  **Model Evaluation:** Assessing the model's accuracy on unseen data, achieving an accuracy of **~81%**.
5.  **Interactive Prediction:** A user-friendly function that allows anyone to input passenger details and receive a real-time survival probability estimate.

---

## 3. Technologies Used

* **Python**
* **Pandas:** For data manipulation and analysis.
* **NumPy:** For numerical operations.
* **Matplotlib & Seaborn:** For data visualization (used during development).
* **Scikit-learn:** For building and evaluating the machine learning model.
* **Jupyter Notebook:** For documenting the entire process.

---

## 4. How to Use

1.  **Explore the Analysis:** Open the `titanic_survival_prediction.ipynb` file directly in GitHub to see the complete, step-by-step analysis.
2.  **Run it Yourself:**
    * Clone this repository.
    * Ensure you have the required libraries installed (`pip install pandas numpy scikit-learn jupyter`).
    * Launch the notebook and run the cells to see the process live or test the interactive predictor with your own data.
