import streamlit as st
import pandas as pd
import pickle

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Logistics Demand Predictor",
    page_icon="🚚",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>

.stApp {
    background: #071426;
    color: white;
}

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
}

/* Header */
.header {
    background: linear-gradient(135deg, #102b52, #0b1d35);
    padding: 28px 32px;
    border-radius: 16px;
    border: 1px solid #1d5fa7;
    margin-bottom: 22px;
}

.header h1 {
    margin: 0;
    font-size: 30px;
    color: white;
}

.header p {
    margin-top: 8px;
    color: #9fb5d1;
    font-size: 15px;
}

/* Model badge */
.model-box {
    background: #0d213b;
    border: 1px solid #245b91;
    padding: 18px 22px;
    border-radius: 12px;
    margin-bottom: 25px;
}

.model-label {
    color: #8da7c5;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.model-name {
    color: #4da3ff;
    font-size: 19px;
    font-weight: 600;
    margin-top: 5px;
}

/* Section */
.section-title {
    font-size: 19px;
    font-weight: 600;
    margin: 10px 0 15px 0;
    color: white;
}

/* Inputs */
div[data-testid="stTextInput"] input,
div[data-testid="stSelectbox"] div {
    background-color: #101f33 !important;
    color: white !important;
    border: 1px solid #294766 !important;
    border-radius: 8px !important;
}

label {
    color: #b9c9dc !important;
}

/* Button */
.stButton > button {
    width: 100%;
    height: 48px;
    background: #1976d2;
    color: white;
    border: none;
    border-radius: 9px;
    font-size: 16px;
    font-weight: 600;
}

.stButton > button:hover {
    background: #2588ed;
}

/* Result */
.result-box {
    background: linear-gradient(135deg, #102f4f, #0b2037);
    border: 1px solid #2878ba;
    border-radius: 14px;
    padding: 28px;
    text-align: center;
    margin-top: 25px;
}

.result-label {
    color: #9eb8d3;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.result-value {
    color: #55b6ff;
    font-size: 42px;
    font-weight: 700;
    margin: 8px 0;
}

.result-unit {
    color: #9eb8d3;
    font-size: 14px;
}

/* Footer */
.footer {
    text-align: center;
    color: #637b96;
    font-size: 12px;
    margin-top: 35px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
with open("logistics_best_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("logistics_scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("logistics_features.pkl", "rb") as file:
    feature_columns = pickle.load(file)

df = pd.read_csv("logistics_dataset-selected-columns.csv")

categories = sorted(
    df["category"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)


# --------------------------------------------------
# MODEL NAME
# --------------------------------------------------
model_name = type(model).__name__

model_display_names = {
    "LinearRegression": "Linear Regression",
    "SVR": "Support Vector Regression",
    "DecisionTreeRegressor": "Decision Tree Regression",
    "RandomForestRegressor": "Random Forest Regression",
    "GradientBoostingRegressor": "Gradient Boosting Regression"
}

display_model = model_display_names.get(
    model_name,
    model_name
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
<div class="header">
    <h1>🚚 Logistics Daily Demand Prediction</h1>
    <p>
        Machine Learning based demand prediction system
        for logistics and inventory management.
    </p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# MODEL
# --------------------------------------------------
st.markdown(f"""
<div class="model-box">
    <div class="model-label">Active Machine Learning Model</div>
    <div class="model-name">⚙️ {display_model}</div>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------
st.markdown(
    '<div class="section-title">📦 Item Details</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    category = st.selectbox(
        "Category",
        categories
    )

    stock_level_text = st.text_input(
        "Stock Level",
        value="50"
    )

    reorder_point_text = st.text_input(
        "Reorder Point",
        value="20"
    )

    reorder_frequency_text = st.text_input(
        "Reorder Frequency (Days)",
        value="7"
    )


with col2:

    lead_time_text = st.text_input(
        "Lead Time (Days)",
        value="5"
    )

    demand_std_text = st.text_input(
        "Demand Standard Deviation",
        value="5"
    )

    popularity_text = st.text_input(
        "Item Popularity Score",
        value="50"
    )

    storage_text = st.text_input(
        "Storage Location ID",
        value="1"
    )


# --------------------------------------------------
# PREDICT BUTTON
# --------------------------------------------------
st.write("")

predict_button = st.button(
    "🔮 Predict Daily Demand"
)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if predict_button:

    try:

        stock_level = float(stock_level_text)
        reorder_point = float(reorder_point_text)
        reorder_frequency_days = float(reorder_frequency_text)
        lead_time_days = float(lead_time_text)
        demand_std_dev = float(demand_std_text)
        item_popularity_score = float(popularity_text)
        storage_location_id = float(storage_text)

        input_data = pd.DataFrame({
            "stock_level": [stock_level],
            "reorder_point": [reorder_point],
            "reorder_frequency_days": [reorder_frequency_days],
            "lead_time_days": [lead_time_days],
            "demand_std_dev": [demand_std_dev],
            "item_popularity_score": [item_popularity_score],
            "storage_location_id": [storage_location_id],
            "category": [category]
        })

        # Encode category
        input_data = pd.get_dummies(
            input_data,
            columns=["category"],
            drop_first=True
        )

        # Match training columns
        input_data = input_data.reindex(
            columns=feature_columns,
            fill_value=0
        )

        # Prediction
        if model_name in [
            "LinearRegression",
            "SVR"
        ]:
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)
        else:
            prediction = model.predict(input_data)

        predicted_value = float(prediction[0])

        # --------------------------------------------------
        # RESULT
        # --------------------------------------------------
        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">Predicted Daily Demand</div>
            <div class="result-value">{predicted_value:.2f}</div>
            <div class="result-unit">units per day</div>
        </div>
        """, unsafe_allow_html=True)

    except ValueError:

        st.error(
            "⚠️ Please enter valid numeric values."
        )

    except Exception as e:

        st.error(
            f"Prediction error: {e}"
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("""
<div class="footer">
    Logistics Demand Prediction System • Machine Learning Project
</div>
""", unsafe_allow_html=True)