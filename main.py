import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt #type: ignore
import seaborn as sns #type: ignore
import webbrowser
from scipy import stats #type: ignore
from sklearn.feature_selection import f_classif #type: ignore
from sklearn.model_selection import train_test_split #type: ignore
from sklearn.linear_model import LogisticRegression #type: ignore
from sklearn.metrics import accuracy_score #type: ignore

# URL do arquivo raw do dataset Titanic no GitHub
url_titanic = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

# Importa o dataset diretamente da URL
df = pd.read_csv(url_titanic)
# print(df.describe())
# print(df.info())

# Calculate correlations with Survived
# numerical_features = ['Age', 'Fare', 'SibSp', 'Parch']
# correlations = df[numerical_features + ['Survived']].corr()['Survived'].sort_values(ascending=False)
# print(correlations)

# Handle missing values in 'Age' column
df['Age'].fillna(df['Age'].median(), inplace=True)  # Impute with median
# Bin Age into groups
age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80]
age_labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80']
df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)

# Encode categorical variables
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['AgeGroup'] = df['AgeGroup'].astype('category').cat.codes  # Convert to numerical codes
# Select relevant features using ANOVA F-value
X = df[['Pclass', 'Sex', 'Age', 'Fare']]
y = df['Survived']
f_values, p_values = f_classif(X, y)

# Create a DataFrame to compare feature importance
# feature_importance = pd.DataFrame({
#     'Feature': X.columns,
#     'F-value': f_values,
#     'P-value': p_values
# })
# print(feature_importance.sort_values('F-value', ascending=False))

# Select features and target
X = df[['Sex', 'Pclass', 'Fare', 'AgeGroup']]
y = df['Survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("TITANIC SURVIVAL PREDICTION\nModel Accuracy:", accuracy_score(y_test, y_pred))

# Interactive prediction
def get_user_input():
    print("Enter your details to estimate survival chances:")
    name = input("Name: ").lower()
    sex = input("Sex (male/female): ").lower()
    pclass = int(input("Passenger Class (1, 2, or 3): "))
    fare = float(input("Fare paid - from $1 to $500): "))
    age = float(input("Age (e.g., 25): "))
    
    # Encode user input
    sex_encoded = 1 if sex == 'female' else 0
    age_group = pd.cut([age], bins=age_bins, labels=age_labels, include_lowest=True)[0]
    age_group_encoded = pd.Categorical([age_group], categories=age_labels).codes[0]
    
    return name, [sex_encoded, pclass, fare, age_group_encoded]

# Get user input and predict
user_name, user_data = get_user_input()
survival_prob = model.predict_proba([user_data])[0][1]
print(user_name.upper(), f", YOUR ESTIMATED SURVIVAL CHANCE: {survival_prob:.2%}")