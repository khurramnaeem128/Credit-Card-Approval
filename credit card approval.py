import pandas as pd

application = pd.read_csv("application_record.csv")
credit = pd.read_csv("credit_record.csv")

print(application.head().to_string())
print(credit.head().to_string())

print("Application Record:")
print(application.shape)
print(application.info())
print(application.isnull().sum())
print(application.duplicated().sum())

print("\nCredit Record:")
print(credit.shape)
print(credit.info())
print(credit.isnull().sum())
print(credit.duplicated().sum())

print("Credit Status Values:")
print(credit["STATUS"].value_counts())

print("\nUnique Status Values:")
print(credit["STATUS"].unique())

print("\nUnique IDs in Application:")
print(application["ID"].nunique())

print("\nUnique IDs in Credit Record:")
print(credit["ID"].nunique())

print("\nCommon IDs:")
print(len(set(application["ID"]) & set(credit["ID"])))

# Convert credit status into numeric values
status_map = {
    "C": 0,
    "X": 0,
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5
}

credit["STATUS_NUM"] = credit["STATUS"].map(status_map)

print("\nCredit Record with Numeric Status:")
print(credit.head())

# Find the highest overdue status for each customer
credit_summary = credit.groupby("ID")["STATUS_NUM"].max().reset_index()

print("\nCredit Summary:")
print(credit_summary.head())

print("\nCredit Summary Shape:")
print(credit_summary.shape)

# Create credit risk target
credit_summary["Credit_Risk"] = credit_summary["STATUS_NUM"].apply(
    lambda x: 1 if x >= 3 else 0
)

print("\nCredit Risk:")
print(credit_summary.head())

print("\nCredit Risk Counts:")
print(credit_summary["Credit_Risk"].value_counts())

# Merge application data with credit risk
df = application.merge(credit_summary, on="ID", how="inner")

print("\nMerged Dataset:")
print(df.head().to_string())

print("\nMerged Dataset Shape:")
print(df.shape)

print("\nCredit Risk Counts After Merge:")
print(df["Credit_Risk"].value_counts())

# Remove duplicate rows
df = df.drop_duplicates()

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing occupation values
df["OCCUPATION_TYPE"] = df["OCCUPATION_TYPE"].fillna("Unknown")

# Separate features and target
X = df.drop(["Credit_Risk", "STATUS_NUM", "ID"], axis=1)
y = df["Credit_Risk"]

print("\nX Shape:")
print(X.shape)

print("\ny Shape:")
print(y.shape)

print(X.select_dtypes(include="str").columns)

# Convert categorical columns into numerical columns
X = pd.get_dummies(X, columns=[
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE"
], dtype=int)

print("\nEncoded X:")
print(X.head())

print("\nEncoded X Shape:")
print(X.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining X Shape:")
print(X_train.shape)

print("\nTesting X Shape:")
print(X_test.shape)

print("\nTraining y Shape:")
print(y_train.shape)

print("\nTesting y Shape:")
print(y_test.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaling completed!")


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train_scaled, y_train)

print("Model trained successfully!")

# Make predictions
y_pred = model.predict(X_test_scaled)

print("\nPredicted Credit Risk:")
print(y_pred[:20])

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)

model_rf.fit(X_train, y_train)

print("Random Forest model trained successfully!")

y_pred_rf = model_rf.predict(X_test)

print("\nRandom Forest Predictions:")
print(y_pred_rf[:20])


print("\nRandom Forest Accuracy:")
print(accuracy_score(y_test, y_pred_rf))

print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

print("\n========== MODEL COMPARISON ==========")

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

final_prediction = pd.DataFrame({
    "Actual_Risk": y_test.values,
    "Predicted_Risk": y_pred_rf
})

final_prediction["Predicted_Risk"] = final_prediction["Predicted_Risk"].map({
    0: "Low Risk",
    1: "High Risk"
})

final_prediction["Actual_Risk"] = final_prediction["Actual_Risk"].map({
    0: "Low Risk",
    1: "High Risk"
})

print("\nFinal Credit Risk Predictions:")
print(final_prediction.head(20))

print("\n========== FINAL PROJECT RESULT ==========")

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

print("\nFinal Conclusion:")
print("Random Forest achieved higher overall accuracy than Logistic Regression.")
print("However, Logistic Regression achieved higher recall for High Risk customers.")
print("Because the dataset is highly imbalanced, accuracy alone is not enough to select the best model.")
print("Both models were evaluated using accuracy, precision, recall, and F1-score.")