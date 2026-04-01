# ## Phase 1: Exploratory Data Analysis (EDA)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



import warnings
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")



df = pd.read_csv("AIML Dataset.csv")
df.head()



df.info()



df.columns



df["isFraud"].value_counts()



df["isFlaggedFraud"].value_counts()



df.isnull().sum()



df.isnull().sum().sum()



df.shape



df.shape[0]  # number of rows and all samples



percentage_of_fraud = (df["isFraud"].value_counts()[1] / df.shape[0]) * 100
print(f"Percentage of fraudulent transactions: {percentage_of_fraud:.2f}%")



df["type"].value_counts()



df["type"].value_counts().plot(kind="bar")
plt.title("Distribution of Transaction Types")
plt.xlabel("Transaction Type")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()



frauds_by_type = df[df["isFraud"] == 1]["type"].value_counts()
frauds_by_type




frauds_by_type = df[df["isFraud"] == 1]["type"].value_counts()
frauds_by_type.plot(kind="bar")
plt.title("Distribution of Fraudulent Transactions by Type")
plt.xlabel("Transaction Type")
plt.ylabel("Count of Fraudulent Transactions")
plt.xticks(rotation=45)
plt.show()



frauds_by_type = df.groupby("type")["isFraud"].mean().sort_values(ascending=False)
frauds_by_type



frauds_by_type = df.groupby("type")["isFraud"].mean().sort_values(ascending=False)
frauds_by_type.plot(kind="bar")
plt.title("Percentage of Fraudulent Transactions by Type")
plt.xlabel("Transaction Type")
plt.ylabel("Percentage of Fraudulent Transactions")
plt.xticks(rotation=45)
plt.show()



df["amount"].describe().astype(int)



sns.histplot(df["amount"], bins=50, kde=True)
plt.title("Distribution of Transaction Amounts")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.show()



sns.histplot(np.log1p(df["amount"]), bins=50, kde=True) # log transformation to handle skewness
plt.title("Distribution of Log-Transformed Transaction Amounts (log scale)")
plt.xlabel("Log(Amount + 1)")
plt.ylabel("Frequency")
plt.show()



data = df["amount"] < 10000 # filter out transactions with amount less than 10,000
sns.histplot(x=df[data]["amount"], bins=50, kde=True)
plt.title("Distribution of Transaction Amounts (Filtered under 10k)")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.show()



data = df["amount"] < 10000 # filter out transactions with amount less than 10,000
sns.boxplot(data=df[data], x="isFraud", y="amount")
plt.title("Boxplot of Transaction Amounts by Fraud Status (Filtered under 10k)")
plt.xlabel("Is Fraud")
plt.ylabel("Amount")
plt.xticks([0, 1], ["Not Fraud", "Fraud"])
plt.show()



df["balanceDiffOriginal"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
df["balanceDiffDestination"] = df["newbalanceDest"] - df["oldbalanceDest"]



(df["balanceDiffOriginal"]< 0).sum()  # count of transactions where original balance decreased



(df["balanceDiffDestination"] < 0).sum()  # count of transactions where destination balance decreased



df.head()




frauds_by_step = df[df["isFraud"] == 1]["step"].value_counts().sort_index()



plt.plot(frauds_by_step.index, frauds_by_step.values, label="Fraudulent Transactions")
plt.title("Number of Fraudulent Transactions Over Time (by Step)")
plt.xlabel("Step")
plt.ylabel("Number of Fraudulent Transactions")
plt.show()



# No need for step column as it doesn't provide meaningful insights for fraud detection in this dataset.
df = df.drop(columns=["step"])
df.head()



top_senders = df["nameOrig"].value_counts().head(10)
top_senders



top_recipients = df["nameDest"].value_counts().head(10)
top_recipients



# How to get top 10 fraudulent senders and recipients
fraudulent_senders = df[df["isFraud"] == 1]["nameOrig"].value_counts().head(10)
fraudulent_recipients = df[df["isFraud"] == 1]["nameDest"].value_counts().head(10)

print(fraudulent_senders)
fraudulent_recipients



frauds_types = df[df["type"].isin(["TRANSFER", "CASH_OUT"])]



frauds_types["type"].value_counts()



frauds_types = df[df["type"].isin(["TRANSFER", "CASH_OUT"])]["isFraud"].value_counts()

frauds_types



frauds_types = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].groupby(["type", "isFraud"]).size().unstack(fill_value=0)

frauds_types.plot(kind="bar")
plt.title("Fraud Transactions by Type (TRANSFER vs CASH_OUT)")
plt.xlabel("Transaction Type")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.legend(labels=["Not Fraud", "Fraud"])
plt.grid(True, axis="y", linestyle="--", alpha=0.7)
plt.show()



# The following plot shows only the fraudulent transactions for TRANSFER and CASH_OUT types, making the fraud counts visible. Since fraud is rare (only 0.13% of transactions), filtering to show only frauds ensures the bars are prominent.


df["isFraud"].value_counts()



df[df["type"].isin(["TRANSFER", "CASH_OUT"])]["isFraud"].value_counts()



frauds_types = df[(df["type"].isin(["TRANSFER", "CASH_OUT"])) & (df["isFraud"] == 1)].groupby("type").size()

frauds_types.plot(kind="bar")
plt.title("Fraudulent Transactions by Type (TRANSFER vs CASH_OUT)")
plt.xlabel("Transaction Type")
plt.ylabel("Count of Fraudulent Transactions")
plt.xticks(rotation=0)
plt.grid(True, axis="y", linestyle="--", alpha=0.7)
plt.show()



# Correlation between amount and fraud
correlation = df[["amount", "isFraud"]].corr()
print(correlation)



# Correlation Matrix
correlation_matrix = df[["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "isFraud"]].corr()



correlation_matrix



sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Key Features")
plt.show()



zero_balance_after_transaction = df[(df["oldbalanceOrg"] > 0) & (df["newbalanceOrig"] == 0)]
zero_balance_after_transaction = zero_balance_after_transaction[zero_balance_after_transaction["type"].isin(["TRANSFER", "CASH_OUT"])]

print(f"Number of transactions where original balance goes to zero after transaction: {zero_balance_after_transaction.shape[0]}")



zero_balance_after_transaction.head()


