import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from openai import OpenAI
import utils as ut


# Initialize the OpenAI client using the correct environment variable
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get('GROQ_API_KEY')  # Ensure this matches the secrets key you've set up
)

# Function to load the machine learning models from .pkl files
def load_model(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)

# Load all models
xgboost_model = load_model('xgb_model.pkl')
native_bayes_model = load_model('nb_model.pkl')
random_forest_model = load_model('rf_model.pkl')
decision_tree_model = load_model('dt_model.pkl')
svm_model = load_model('SVM_model.pkl')
knn_model = load_model('knn_model.pkl')
voting_classifier_model = load_model('voting_clf.pkl')
xgboost_SMOTE_model = load_model('xgboost-SMOTE.pkl')
xgboost_featureEngineered_model = load_model('xgboost-featureEngineered.pkl')

# Prepare input for the model
def prepare_input(credit_score, location, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary):
    input_dict = {
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': int(has_credit_card),
        'IsActiveMember': int(is_active_member),
        'EstimatedSalary': estimated_salary,
        'Geography_France': 1 if location == 'France' else 0,
        'Geography_Germany': 1 if location == 'Germany' else 0,
        'Geography_Spain': 1 if location == 'Spain' else 0,
        'Gender_Male': 1 if gender == 'Male' else 0,
        'Gender_Female': 1 if gender == 'Female' else 0
    }

    input_df = pd.DataFrame([input_dict])
    return input_df, input_dict


def make_predictions(input_df, input_dict):
    input_df = input_df[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
                         'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
                         'Geography_France', 'Geography_Germany', 'Geography_Spain',
                         'Gender_Female', 'Gender_Male']]

    probabilities = {
        'XGBoost': xgboost_model.predict_proba(input_df)[0][1],
        'Random Forest': random_forest_model.predict_proba(input_df)[0][1],
        'K-Nearest Neighbors': knn_model.predict_proba(input_df)[0][1]
    }

    avg_probability = np.mean(list(probabilities.values()))

    col1, col2 = st.columns(2)

    # Within the col1 and col2 sections in make_predictions function:
    with col1:
        fig = ut.create_gauge_chart(avg_probability)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})  # Disable toolbar
        st.write(f"The customer has a {avg_probability:.2%} probability of churning.")

    with col2:
        fig_probs = ut.create_model_probability_chart(probabilities)
        st.plotly_chart(fig_probs, use_container_width=True, config={'displayModeBar': False})  # Disable toolbar

    return avg_probability

def explain_prediction(probability, input_dict, surname):
    prompt = f"""
    You are an expert data scientist at a bank. You specialize in interpreting and explaining predictions of machine learning models, particularly customer churn predictions. 
    Your task is to explain the following prediction to a non-technical customer service representative.

    Your machine learning model has predicted that a customer named {surname} has a {round(probability * 100, 1)}% probability of churning based on the information provided below.

    Please provide:
    1. **An overview** of the prediction, including why the customer might churn.
    2. **A detailed breakdown** of the key features (from the model's top 10 most important features) that most contribute to this prediction.
    3. **Actionable advice** for the customer service representative on how to reduce the churn risk based on the provided features.

    Here's the customer's information:
    {input_dict}

    Here are the top 10 most important features from the machine learning model, ranked by importance:

    Feature | Importance
    ----------------------
    NumOfProducts    | 0.323888
    IsActiveMember   | 0.164146
    Age              | 0.109559
    Geography_Germany| 0.095376
    Balance          | 0.057875
    Geography_France | 0.052467
    Gender_Female    | 0.049893
    EstimatedSalary  | 0.031940
    HasCrCard        | 0.030954
    Tenure           | 0.030054
    Gender_Male      | 0.000000

    Based on this, generate a detailed 3-sentence explanation of why {surname} might churn, as well as suggestions to reduce churn risk.
    """

    raw_response = client.chat.completions.create(
        model="llama-3.2-3b-preview",  # Replace with your actual model name
        messages=[{
            "role": "user",
            "content": prompt
        }],
    )

    return raw_response.choices[0].message.content


def generate_email(probability, input_dict, explanation, surname):
    prompt = f"""
    You are a relationship manager at PAK Bank, responsible for ensuring customer satisfaction and retention. Your goal is to craft a personalized and professional email to a valued customer, {surname}, offering services that address their current needs and encourage continued loyalty.

    We have identified that {surname} has a {round(probability * 100, 1)}% chance of churning based on recent analysis. As a proactive measure, we would like to extend our appreciation for their loyalty to the bank and offer tailored incentives to ensure their satisfaction with our services.

    Customer Information:
    {input_dict}

    Key reasons {surname} may be at risk of churning:
    {explanation}

    In your email:
    1. **Appreciate their loyalty** without being overly personal. Acknowledge their long-standing relationship with the bank.
    2. **Offer relevant incentives** that could benefit the customer, such as exclusive promotions, improved interest rates, or enhanced services based on their financial profile.
    3. **Encourage a conversation** to understand their needs better and provide reassurance that the bank is committed to their financial well-being.
    4. **Maintain a professional tone** that balances appreciation and offers while avoiding any emotional or overly personal language.

    Example incentives to include:
    - Exclusive interest rates on savings or loans.
    - Enhanced rewards programs or banking services.
    - Dedicated support to help them manage their financial goals.

    Avoid mentioning the predicted probability of churning or the use of machine learning models in the communication. Focus on the practical benefits of staying with the bank and provide clear, actionable incentives to retain their business.

    """

    raw_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # Replace with the actual model you're using
        messages=[{
            "role": "user",
            "content": prompt
        }],
    )

    print("\n\nEMAIL PROMPT", prompt)

    return raw_response.choices[0].message.content





st.title("Customer Churn Prediction")

df = pd.read_csv("churn.csv")

customers = [f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()]

selected_customer_option = st.selectbox("Select a customer", customers)

if selected_customer_option:
    selected_customer_id = int(selected_customer_option.split(" - ")[0])

selected_surname = selected_customer_option.split(" - ")[1]

selected_customer = df.loc[df['CustomerId'] == selected_customer_id].iloc[0]

col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=int(selected_customer['CreditScore'])
    )

    location = st.selectbox(
        "Location", ["Spain", "France", "Germany"],
        index=["Spain", "France", "Germany"].index(selected_customer['Geography'])
    )

    gender = st.radio("Gender", ["Male", "Female"],
                      index=0 if selected_customer['Gender'] == 'Male' else 1)

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=int(selected_customer['Age'])
    )

    tenure = st.number_input(
        "Tenure (years)",
        min_value=0,
        max_value=50,
        value=int(selected_customer['Tenure'])
    )

with col2:
    balance = st.number_input(
        "Balance",
        min_value=0.0,
        value=float(selected_customer['Balance'])
    )

    num_products = st.number_input(
        "Number of Products",
        min_value=1,
        max_value=10,
        value=int(selected_customer['NumOfProducts'])
    )

    has_credit_card = st.checkbox(
        "Has Credit Card",
        value=bool(selected_customer['HasCrCard'])
    )

    is_active_member = st.checkbox(
        "Is Active Member",
        value=bool(selected_customer['IsActiveMember'])
    )

    estimated_salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        value=float(selected_customer['EstimatedSalary'])
    )


input_df, input_dict = prepare_input(credit_score, location, gender, age, tenure, balance,
                                     num_products, has_credit_card, is_active_member, estimated_salary)

avg_probability =  make_predictions(input_df, input_dict)


explanation = explain_prediction(avg_probability, input_dict, selected_customer['Surname'])

st.markdown("---")
st.subheader("Explanation of Prediction")
st.markdown(explanation)


email = generate_email(avg_probability, input_dict, explanation, selected_customer['Surname'])

st.markdown("---")

st.subheader("Personalized Email")

st.markdown(email)
