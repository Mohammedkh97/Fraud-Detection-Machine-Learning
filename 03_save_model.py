# ## Phase 3: Save the Trained Model


# Save the model
import joblib

joblib.dump(model_pipeline, "lr_fraud_model.pkl")
joblib.dump(rf_pipeline, "rf_fraud_model.pkl")
joblib.dump(xgb_pipeline, "xgb_fraud_model.pkl")



# Test the model on a single transaction

trained_model = joblib.load("xgb_fraud_model.pkl")
sample_transaction = X_test.iloc[0:1]  # Take the first transaction from the test set
predicted_label = trained_model.predict(sample_transaction)
print(f"Predicted label for the sample transaction: {predicted_label[0]}")


