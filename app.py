import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="ML Prediction Systems", page_icon="📊", layout="centered")
ARTIFACTS = Path("artifacts")

@st.cache_resource
def load_models():
    t_path = ARTIFACTS / "titanic_model.joblib"
    h_path = ARTIFACTS / "heart_disease_model.joblib"
    if not t_path.exists() or not h_path.exists():
        return None, None
    return joblib.load(t_path), joblib.load(h_path)

titanic_model, heart_model = load_models()
st.title("End-to-End Machine Learning Systems")
st.write("Educational demonstrations for Titanic survival and heart disease classification.")

if titanic_model is None or heart_model is None:
    st.error("Model files are missing. Run the Jupyter notebook first to create artifacts/titanic_model.joblib and artifacts/heart_disease_model.joblib, then deploy the app with the artifacts folder.")
    st.stop()

tab1, tab2 = st.tabs(["Titanic Survival", "Heart Disease"])

with tab1:
    st.subheader("Titanic Passenger Survival Prediction")
    pclass = st.selectbox("Passenger class", [1, 2, 3], index=2)
    sex = st.selectbox("Sex", ["female", "male"])
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=28.0)
    sibsp = st.number_input("Siblings/spouses aboard", min_value=0, max_value=10, value=0)
    parch = st.number_input("Parents/children aboard", min_value=0, max_value=10, value=0)
    fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=12.0)
    embarked = st.selectbox("Port of embarkation", ["S", "C", "Q"])
    alone = st.checkbox("Travelling alone", value=True)
    if st.button("Predict survival", key="titanic_predict"):
        row = pd.DataFrame([{"pclass": pclass, "sex": sex, "age": age, "sibsp": sibsp,
                             "parch": parch, "fare": fare, "embarked": embarked, "alone": alone}])
        pred = int(titanic_model.predict(row)[0])
        prob = float(titanic_model.predict_proba(row)[0, 1])
        st.success("Predicted outcome: " + ("Survived" if pred else "Did not survive"))
        st.metric("Estimated survival probability", f"{prob:.1%}")

with tab2:
    st.subheader("Heart Disease Prediction")
    st.warning("Educational model only — not medical advice or a diagnostic device.")
    age_h = st.number_input("Age (years)", min_value=1, max_value=120, value=55, key="age_h")
    sex_h = st.selectbox("Sex code (UCI: 1=male, 0=female)", [1, 0])
    cp = st.selectbox("Chest pain type (UCI code)", [1, 2, 3, 4], index=1)
    thalach = st.number_input("Maximum heart rate achieved (thalach)", min_value=50, max_value=250, value=150)
    if st.button("Estimate model class", key="heart_predict"):
        row = pd.DataFrame([{"age": age_h, "sex": sex_h, "cp": cp, "thalach": thalach}])
        pred = int(heart_model.predict(row)[0])
        prob = float(heart_model.predict_proba(row)[0, 1])
        st.write("Model output:", "**Disease present**" if pred else "**No disease detected by model**")
        st.metric("Model-estimated probability of positive class", f"{prob:.1%}")
        st.caption("This output must not be used to make health decisions. Consult a qualified healthcare professional.")
