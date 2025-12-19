import streamlit as st
import pandas as pd
import joblib

# ===============================
# Load trained pipeline
# ===============================
MODEL_PATH = r"C:\Users\AB\Desktop\tour_package_recommanding_system\model\product_pitch_pipeline.pkl"
pipeline = joblib.load(MODEL_PATH)
MODEL_CLASSES = pipeline.named_steps["model"].classes_

# ===============================
# Page config
# ===============================
st.set_page_config(
    page_title="Product Pitch Suggestion",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Product Pitch Suggestion App")
st.markdown("Predict the **best travel product to pitch** using Machine Learning.")
st.divider()

# ===============================
# Feature schema (MUST match backend)
# ===============================
FEATURE_COLUMNS = [
    'Age', 'TypeofContact', 'CityTier', 'DurationOfPitch',
    'Occupation', 'Gender', 'NumberOfPersonVisiting',
    'NumberOfFollowups', 'PreferredPropertyStar',
    'MaritalStatus', 'NumberOfTrips', 'Passport',
    'PitchSatisfactionScore', 'OwnCar',
    'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome'
]

# ============================================================
# 🔹 SINGLE PREDICTION
# ============================================================
st.header("🔹 Single Customer Prediction")

col1, col2 = st.columns(2)

with col1:
    Age = st.slider("Age", 18, 100, 30)
    TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    CityTier = st.selectbox("City Tier", [1, 2, 3])
    DurationOfPitch = st.slider("Duration of Pitch (minutes)", 1, 60, 10)
    Occupation = st.selectbox(
        "Occupation",
        ["Free Lancer", "Salaried", "Small Business", "Large Business"]
    )
    Gender = st.selectbox("Gender", ["Male", "Female"])

with col2:
    NumberOfPersonVisiting = st.slider("Number of Persons Visiting", 1, 20, 2)
    NumberOfFollowups = st.slider("Number of Follow-ups", 0, 20, 1)
    PreferredPropertyStar = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
    MaritalStatus = st.selectbox(
        "Marital Status",
        ["Single", "Unmarried", "Divorced", "Married"]
    )
    NumberOfTrips = st.slider("Number of Trips", 0, 20, 1)
    Passport = st.selectbox("Has Passport?", [0, 1])
    PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    OwnCar = st.selectbox("Owns a Car?", [0, 1])
    NumberOfChildrenVisiting = st.slider("Number of Children Visiting", 0, 10, 0)
    Designation = st.selectbox(
        "Designation",
        ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )
    MonthlyIncome = st.slider("Monthly Income", 0, 1_000_000, 20_000, step=1000)

# Prepare dataframe for prediction
customer_df = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome
}])

if st.button("🎯 Suggest Product", use_container_width=True):
    try:
        pred = pipeline.predict(customer_df)[0]
        proba = pipeline.predict_proba(customer_df)[0]

        st.success(f"✅ **Recommended Product:** `{pred}`")

        st.subheader("📊 Prediction Confidence")
        for cls, p in zip(MODEL_CLASSES, proba):
            st.write(f"**{cls}** : {p * 100:.2f}%")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ============================================================
# 🔹 BULK PREDICTION
# ============================================================
st.divider()
st.header("🔹 Bulk Prediction (CSV Upload)")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]

    if missing:
        st.error(f"❌ Missing required columns: {missing}")
    else:
        st.dataframe(df.head())
        st.info(f"Rows uploaded: {len(df)}")

        if st.button("🚀 Run Bulk Prediction", use_container_width=True):
            try:
                preds = pipeline.predict(df[FEATURE_COLUMNS])
                probs = pipeline.predict_proba(df[FEATURE_COLUMNS])

                result = df.copy()
                result["Recommended_Product"] = preds

                for i, cls in enumerate(MODEL_CLASSES):
                    result[f"Prob_{cls.replace(' ', '_')}"] = probs[:, i]

                st.success("✅ Bulk prediction completed")
                st.dataframe(result)

                st.download_button(
                    "⬇️ Download Predictions",
                    result.to_csv(index=False),
                    file_name="product_pitch_predictions.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Bulk prediction failed: {e}")
