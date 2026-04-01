import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb

# Set up page configuration
st.set_page_config(
    page_title="Fraud Detection AI",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS for a premium look
st.markdown("""
<style>
    .main-header {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-family: 'Inter', sans-serif;
        color: #475569;
        text-align: center;
        margin-bottom: 3rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #2563EB;
        color: white;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.6rem;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    # Load the XGBoost pipeline
    try:
        model = joblib.load("xgb_fraud_model.pkl")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    st.markdown("<h1 class='main-header'>🛡️ Fraud Detection AI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Analyze transactions in real-time using our advanced XGBoost machine learning pipeline.</p>", unsafe_allow_html=True)

    model = load_model()

    if model is None:
        st.warning("Please ensure 'xgb_fraud_model.pkl' is present in the application directory.")
        return

    # Transaction Form
    with st.container():
        st.subheader("Transaction Details")
        col1, col2 = st.columns(2)

        with col1:
            txn_type = st.selectbox(
                "Transaction Type",
                options=["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"],
                help="Select the classification of this transaction."
            )
            amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=1000.0, step=100.0)
            
        with col2:
            st.info("Provide balance details to calculate origin and destination differences automatically.")

        st.markdown("---")
        
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("#### Sender (Origin)")
            oldbalanceOrg = st.number_input("Old Balance", min_value=0.0, value=5000.0, step=100.0, key='oldOrg')
            newbalanceOrig = st.number_input("New Balance", min_value=0.0, value=4000.0, step=100.0, key='newOrg')

        with col4:
            st.markdown("#### Recipient (Destination)")
            oldbalanceDest = st.number_input("Old Balance", min_value=0.0, value=0.0, step=100.0, key='oldDest')
            newbalanceDest = st.number_input("New Balance", min_value=0.0, value=1000.0, step=100.0, key='newDest')

    st.markdown("<br>", unsafe_allow_html=True)

    # Prediction Action
    if st.button("Analyze Transaction"):
        with st.spinner("Analyzing transaction patterns..."):
            # Calculate derived features
            balanceDiffOriginal = oldbalanceOrg - newbalanceOrig
            balanceDiffDestination = newbalanceDest - oldbalanceDest

            # Construct DataFrame exactly matching model training
            input_data = pd.DataFrame([{
                "amount": amount,
                "oldbalanceOrg": oldbalanceOrg,
                "newbalanceOrig": newbalanceOrig,
                "oldbalanceDest": oldbalanceDest,
                "newbalanceDest": newbalanceDest,
                "balanceDiffOriginal": balanceDiffOriginal,
                "balanceDiffDestination": balanceDiffDestination,
                "type": txn_type
            }])

            # Make Prediction
            try:
                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data)[0][1]

                st.markdown("---")
                st.subheader("Analysis Results")
                
                # Display results with attractive callouts
                if prediction == 1:
                    st.error(f"🚨 **FRAUDULENT TRANSACTION DETECTED** 🚨\n\n**Confidence Score:** {probability:.2%}")
                    st.markdown("> *The system has flagged this transaction as highly suspicious based on known fraud patterns. Please review immediately.*")
                else:
                    st.success(f"✅ **TRANSACTION APPROVED** \n\n**Fraud Probability:** {probability:.2%}")
                    st.markdown("> *This transaction appears normal and matches safe transaction patterns.*")
                    
                # Small expander for debugging / feature view
                with st.expander("View Calculated Features"):
                    st.json({
                        "Origin Balance Diff": balanceDiffOriginal,
                        "Destination Balance Diff": balanceDiffDestination
                    })

            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")


if __name__ == "__main__":
    main()
