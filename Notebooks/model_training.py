import pandas as pd

# 1. Import dataset (use relative path for GitHub compatibility)
# Make sure the CSV file is in the same folder as this Python script
df = pd.read_csv('Depression_Student_Dataset.csv')

print(df.head())

# 2. Remove duplicate rows
df.drop_duplicates(inplace=True)

# 3. Handle missing values (if any)
df.dropna(inplace=True)

# 4. Convert categorical columns into numerical values

# Gender: Male=0, Female=1
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Sleep Duration category mapping
sleep_map = {
    'Less than 5 hours': 0,
    '5-6 hours': 1,
    '7-8 hours': 2,
    'More than 8 hours': 3
}
df['Sleep Duration'] = df['Sleep Duration'].map(sleep_map)

# Dietary Habits: Unhealthy=0, Moderate=1, Healthy=2
diet_map = {'Unhealthy': 0, 'Moderate': 1, 'Healthy': 2}
df['Dietary Habits'] = df['Dietary Habits'].map(diet_map)

# Suicidal Thoughts: Yes=1, No=0
df['Have you ever had suicidal thoughts ?'] = df['Have you ever had suicidal thoughts ?'].map({'Yes': 1, 'No': 0})

# Family Mental Illness History: Yes=1, No=0
df['Family History of Mental Illness'] = df['Family History of Mental Illness'].map({'Yes': 1, 'No': 0})

# Target label (Depression): Yes=1, No=0
df['Depression'] = df['Depression'].map({'Yes': 1, 'No': 0})

# 5. Convert all columns to integer type
df = df.astype(int)

# Final information check
df.info()


# ==============================
# MACHINE LEARNING SECTION
# ==============================

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Split features (X) and target label (y)
X = df.drop('Depression', axis=1)
y = df['Depression']

# 2. Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Decision Tree Model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))
print("Decision Tree Report:\n", classification_report(y_test, dt_pred))

# 4. Random Forest Model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("Random Forest Report:\n", classification_report(y_test, rf_pred))
