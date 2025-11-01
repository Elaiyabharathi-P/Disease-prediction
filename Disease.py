import pandas as pd
import streamlit as st

# Load dataset
dataset_path = "dataset.csv"  # Ensure this is the correct path
df = pd.read_csv(dataset_path)

# Function to predict disease
def predict_disease(symptoms):
    symptoms_set = set(symptoms.lower().split(","))
    best_match = None
    max_matches = 0

    for _, row in df.iterrows():
        disease_symptoms = set(row['symptoms'].lower().split(","))
        match_count = len(symptoms_set.intersection(disease_symptoms))

        if match_count > max_matches:
            max_matches = match_count
            best_match = row

    if best_match is not None:
        return {
            "Disease": best_match['disease'],
            "Possible Treatments": best_match['cures'],
            "Recommended Doctor": best_match['doctor'],
            "Risk Level": best_match['risk level']
        }
    return None

# Streamlit UI
st.title("🩺 Disease Prediction System")
st.write("Enter symptoms to predict the most likely disease.")

# User input
user_input = st.text_input("Enter symptoms (comma-separated)", "")

if st.button("Predict Disease"):
    if user_input.strip():
        result = predict_disease(user_input)
        if result:
            st.success(f"**Predicted Disease:** {result['Disease']}")
            st.write(f"**Possible Treatments:** {result['Possible Treatments']}")
            st.write(f"**Recommended Doctor:** {result['Recommended Doctor']}")
            st.write(f"**Risk Level:** {result['Risk Level']}")
        else:
            st.warning("No matching disease found. Consider consulting a doctor.")
    else:
        st.error("Please enter symptoms to proceed.")
