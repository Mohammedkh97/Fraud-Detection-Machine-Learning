# ## Phase 2: Feature Engineering and ML Pipeline


from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer # for handling different types of features (numerical and categorical)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb



df_model = df.copy()
df_model.head()



df_model.drop(columns=["nameOrig", "nameDest", "isFlaggedFraud"], inplace=True)



# ### Feature Engineering


# Types of Data:
# - Numerical: amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, balanceDiffOriginal, balanceDiffDestination
# - Categorical: type

numerical_features = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "balanceDiffOriginal", "balanceDiffDestination"]
categorical_features = ["type"]



y = df_model["isFraud"]
X = df_model.drop(columns=["isFraud"])



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)




# Preprocessing and modeling pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'),categorical_features)
    ],remainder='drop'
)



# ### Train Logistic Regression Model:


model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000))
])



model_pipeline.fit(X_train, y_train)



y_pred = model_pipeline.predict(X_test)



# ### Evaluation Metrices


classification_report(y_test, y_pred)



classification_rep = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_rep)



confusion_matrix(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)



cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
			xticklabels=["Not Fraud", "Fraud"],
			yticklabels=["Not Fraud", "Fraud"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()



# How to calculate and visualize ROC AUC
y_prob = model_pipeline.predict_proba(X_test)[:, 1]  # Get probabilities
print(y_prob[:10])  # Print the first 10 predicted probabilities for the positive class (fraud)
roc_auc = roc_auc_score(y_test, y_prob) # Calculate ROC AUC score (Receiver Operating Characteristic - Area Under the Curve)
print(f"ROC AUC Score: {roc_auc:.4f}")




# Visualize ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob) # Calculate false positive rate, true positive rate, and thresholds for ROC curve
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for reference
plt.title("ROC Curve")



score = model_pipeline.score(X_test, y_test) * 100 
print(f"Model Accuracy: {score:.2f}%")



# ### Random Forest Model:


# Prepare data
# I will use the same X and y from the previous logistic regression model, which are already defined as:

# y = df["isFraud"]
# X = df.drop(columns=["isFraud", "step", "nameOrig", "nameDest"])

numerical_features = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "balanceDiffOriginal", "balanceDiffDestination"]
categorical_features = ["type"]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'),categorical_features)
    ],remainder='drop'
)




# Random Forest
rf_pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1))
])




rf_pipeline.fit(X_train, y_train)



y_pred_rf = rf_pipeline.predict(X_test)
y_proba_rf = rf_pipeline.predict_proba(X_test)[:, 1]



# ### Evaluation Metrices


print("Random Forest classification report:\n", classification_report(y_test, y_pred_rf, digits=4))
print("Random Forest ROC AUC:", roc_auc_score(y_test, y_proba_rf))
print("Random Forest confusion matrix:\n", confusion_matrix(y_test, y_pred_rf))




cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
			xticklabels=["Not Fraud", "Fraud"],
			yticklabels=["Not Fraud", "Fraud"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()



roc_auc_rf = roc_auc_score(y_test, y_proba_rf)
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_proba_rf)
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest ROC Curve (AUC = {roc_auc_rf:.4f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()



score = rf_pipeline.score(X_test, y_test) * 100 
print(f"Model Accuracy: {score:.2f}%")



# ### XGBoost Model:


# Prepare data
# I will use the same X and y from the previous logistic regression model, which are already defined as:

# y = df["isFraud"]
# X = df.drop(columns=["isFraud", "step", "nameOrig", "nameDest"])

numerical_features = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "balanceDiffOriginal", "balanceDiffDestination"]
categorical_features = ["type"]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'),categorical_features)
    ],remainder='drop'
)




# Prepare data
# I will use the same X and y from the previous logistic regression model, which are already defined as:

# XGBoost
xgb_pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
                              n_estimators=200, max_depth=6, random_state=42, n_jobs=-1))
])




xgb_pipeline.fit(X_train, y_train)



y_pred_xgb = xgb_pipeline.predict(X_test)
y_proba_xgb = xgb_pipeline.predict_proba(X_test)[:, 1]

print("XGBoost classification report:\n", classification_report(y_test, y_pred_xgb, digits=4))
print("XGBoost ROC AUC:", roc_auc_score(y_test, y_proba_xgb))
print("XGBoost confusion matrix:\n", confusion_matrix(y_test, y_pred_xgb))



cm = confusion_matrix(y_test, y_pred_xgb)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
			xticklabels=["Not Fraud", "Fraud"],
			yticklabels=["Not Fraud", "Fraud"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()



roc_auc_xgb = roc_auc_score(y_test, y_proba_xgb)
fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(y_test, y_proba_xgb)
plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost ROC Curve (AUC = {roc_auc_xgb:.4f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()



score = xgb_pipeline.score(X_test, y_test) * 100 
print(f"Model Accuracy: {score:.2f}%")


