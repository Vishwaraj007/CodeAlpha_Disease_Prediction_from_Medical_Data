import streamlit as st
import joblib
import numpy as np

# Page config
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🩺",
    layout="centered"
)

# Load model
model = joblib.load("model/diabetes_model.pkl")

# Title section
st.markdown(
    "<h1 style='text-align: center;'>🩺 Diabetes Prediction System</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>AI-powered medical risk prediction</p>",
    unsafe_allow_html=True
)

st.divider()

# Sidebar inputs
st.sidebar.header("🧪 Patient Medical Details")

pregnancies = st.sidebar.slider("🤰 Pregnancies", 0, 20, 1)
glucose = st.sidebar.slider("🩸 Glucose Level", 0, 300, 120)
bp = st.sidebar.slider("💓 Blood Pressure", 0, 200, 70)
skin = st.sidebar.slider("📏 Skin Thickness", 0, 100, 20)
insulin = st.sidebar.slider("💉 Insulin", 0, 900, 80)
bmi = st.sidebar.slider("⚖️ BMI", 0.0, 70.0, 25.0)
dpf = st.sidebar.slider("🧬 Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.sidebar.slider("👤 Age", 1, 120, 30)

# Prediction section
st.subheader("📊 Prediction Result")

if st.button("🔍 Predict Disease"):
    input_data = np.array([[pregnancies, glucose, bp, skin,
                             insulin, bmi, dpf, age]])

    # Progress bar
    with st.spinner("Analyzing patient data..."):
        st.progress(70)

    prediction = model.predict(input_data)

    st.divider()

    if prediction[0] == 1:
        st.error("❌ **High Risk: Diabetes Detected**")
        st.markdown(
            "<p style='color:red;'>⚠️ Please consult a medical professional.</p>",
            unsafe_allow_html=True
        )
    else:
        st.success("✅ **Low Risk: No Diabetes Detected**")
        st.markdown(
            "<p style='color:green;'>🎉 Patient appears healthy.</p>",
            unsafe_allow_html=True
        )

# Footer
st.divider()
st.markdown(
    "<p style='text-align:center; font-size:12px;'>"
    "Built with ❤️ using Python, ML & Streamlit<br>"
    "Internship Project – Disease Prediction"
    "</p>",
    unsafe_allow_html=True
)

