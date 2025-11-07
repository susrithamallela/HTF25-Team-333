import streamlit as st
from PIL import Image
import pandas as pd
import string
from transformers import pipeline
import difflib

# --- Load AI Model ---
@st.cache_resource
def load_model():
    return pipeline("image-classification", model="nateraw/food")  # Public model

model = load_model()

# --- Load Calorie Database ---
FALLBACK_CSV = "food_db.csv"
try:
    fallback_df = pd.read_csv(FALLBACK_CSV)
except Exception as e:
    st.error(f"Error loading food database: {e}")
    st.stop()

# Normalize food names once
fallback_df["normalized_name"] = (
    fallback_df["food_name"]
    .str.lower()
    .str.replace("_", " ")
    .str.translate(str.maketrans("", "", string.punctuation))
    .str.strip()
)

# --- Lookup Function ---
def fallback_lookup(label):
    if not label:
        return None

    main_label = (
        label.lower()
        .replace("_", " ")
        .translate(str.maketrans("", "", string.punctuation))
        .strip()
    )

    # Exact match
    match = fallback_df[fallback_df["normalized_name"] == main_label]
    if not match.empty:
        return float(match.iloc[0]["calories_per_100g"])

    # Partial match
    for _, row in fallback_df.iterrows():
        db_name = row["normalized_name"]
        if db_name in main_label or main_label in db_name:
            return float(row["calories_per_100g"])

    # Fuzzy match
    close = difflib.get_close_matches(main_label, fallback_df["normalized_name"], n=1, cutoff=0.6)
    if close:
        match_row = fallback_df[fallback_df["normalized_name"] == close[0]]
        if not match_row.empty:
            return float(match_row.iloc[0]["calories_per_100g"])

    return 200.0  # fallback calories

# --- Daily Calorie Calculation ---
def calculate_daily_calories(weight_kg, height_cm, age, gender):
    if gender == "male":
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    return bmr * 1.2

# --- Streamlit Config ---
st.set_page_config(page_title="🍔 FoodVision AI Calorie Tracker", layout="centered")

# --- Session Initialization ---
for key, default in {
    "profile_submitted": False,
    "food_history": [],
    "consumed_calories": 0.0,
    "top_label": "",
    "calories_per_100g": 0.0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --- Profile Setup ---
if not st.session_state.profile_submitted:
    st.title("👤 Enter Your Profile")
    username = st.text_input("Name / Username")
    age = st.number_input("Age", min_value=10, max_value=100, value=25)
    height_cm = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
    weight_kg = st.number_input("Weight (kg)", min_value=20, max_value=200, value=70)
    gender = st.selectbox("Gender", ["male", "female"])
    submit_profile = st.button("Submit Profile")

    if submit_profile and username:
        st.session_state.username = username
        st.session_state.daily_calories = calculate_daily_calories(weight_kg, height_cm, age, gender)
        st.session_state.profile_submitted = True
        st.success(f"Profile saved! Daily target: {st.session_state.daily_calories:.0f} kcal")

# --- Main App ---
if st.session_state.profile_submitted:
    st.title(f"Welcome {st.session_state.username}! 👋")
    st.subheader(f"🎯 Daily Calorie Target: {st.session_state.daily_calories:.0f} kcal")
    st.markdown("---")
    st.header("🍽️ Identify Food and Track Calories")

    col1, col2 = st.columns([1, 2])
    with col1:
        uploaded = st.file_uploader("Upload a meal image", type=["jpg", "jpeg", "png"])
        use_camera = st.camera_input("Or take a photo")
        run_btn = st.button("🔍 Analyze Image")

    with col2:
        preview = st.empty()

    # Load image
    if uploaded:
        st.session_state.image = Image.open(uploaded)
    elif use_camera:
        st.session_state.image = use_camera

    if "image" in st.session_state and st.session_state.image is not None:
        preview.image(st.session_state.image, use_container_width=True)

    # Run model
    if run_btn and st.session_state.image is not None:
        with st.spinner("🔎 Identifying dish..."):
            preds = model(st.session_state.image)

        # Top-3 predictions
        top3 = preds[:3]
        options = [p["label"] for p in top3]
        st.write("### 🧠 Top predictions:")
        for p in top3:
            st.write(f"- {p['label']} ({p['score']*100:.1f}% confidence)")

        # Let user select correct dish
        chosen = st.selectbox("Select the correct dish", options)
        st.session_state.top_label = chosen
        st.info(f"✅ You selected: **{chosen}**")

        # Lookup calories
        st.session_state.calories_per_100g = fallback_lookup(chosen)
        st.success(f"Calories found: {st.session_state.calories_per_100g:.0f} kcal per 100g")

    # Calorie Tracking
    if st.session_state.top_label:
        st.markdown("---")
        st.subheader(f"🍛 {st.session_state.top_label}")

        serving_size = st.number_input("Enter serving size (grams)", min_value=10, max_value=2000, value=200, step=10)
        calories_est = st.session_state.calories_per_100g * (serving_size / 100.0)
        st.metric("Estimated Calories for This Serving", f"{calories_est:.0f} kcal")

        add_btn = st.button("Add to Today's Log")
        if add_btn:
            entry = {
                "dish": st.session_state.top_label,
                "serving_g": serving_size,
                "calories": calories_est,
            }
            st.session_state.food_history.append(entry)
            st.session_state.consumed_calories = sum(i["calories"] for i in st.session_state.food_history)
            st.success(f"Added {st.session_state.top_label} ({serving_size}g, {calories_est:.0f} kcal)")

        remaining = max(st.session_state.daily_calories - st.session_state.consumed_calories, 0)
        st.metric("🔥 Total Consumed Calories", f"{st.session_state.consumed_calories:.0f} kcal")
        st.metric("🥗 Remaining Calories You Can Still Consume", f"{remaining:.0f} kcal")

        # Meal History
        if st.session_state.food_history:
            st.markdown("### 📋 Today's Meal History")
            df = pd.DataFrame(st.session_state.food_history)
            st.table(df)

        # Reset
        if st.button("🔄 Reset Today's Log"):
            st.session_state.food_history = []
            st.session_state.consumed_calories = 0.0
            st.success("Log reset successfully!")
