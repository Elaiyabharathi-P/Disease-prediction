import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare the data
def load_data():
    # In a real application, you would load from a CSV file
    # For this example, we'll use the provided data
    data = pd.read_csv('dataset.csv')
    
    # Clean missing values and duplicates
    data = data.dropna(subset=['disease', 'symptoms'])
    data = data.drop_duplicates(subset=['disease'])
    
    # Extract risk level as a numeric value for later use
    data['risk_value'] = data['risk level'].apply(lambda x: 
        float(re.search(r'(\d+(?:\.\d+)?)', str(x)).group(1)) if isinstance(x, str) and re.search(r'(\d+(?:\.\d+)?)', str(x)) else 0)
    
    return data

# Preprocess symptoms data
def preprocess_data(data):
    # Create a dictionary of diseases and their corresponding symptoms
    disease_symptom_dict = {}
    for _, row in data.iterrows():
        disease = row['disease']
        symptoms = row['symptoms'].split(',')
        symptoms = [s.strip() for s in symptoms]
        disease_symptom_dict[disease] = symptoms
    
    # Create a list of all unique symptoms
    all_symptoms = []
    for disease, symptoms in disease_symptom_dict.items():
        all_symptoms.extend(symptoms)
    unique_symptoms = sorted(list(set(all_symptoms)))
    
    return disease_symptom_dict, unique_symptoms

# Build model
def build_model(data, unique_symptoms):
    # Create a feature vector for each disease
    X = []
    y = []
    
    for _, row in data.iterrows():
        disease = row['disease']
        symptom_str = row['symptoms']
        
        # Add to training data
        X.append(symptom_str)
        y.append(disease)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Vectorize the symptom strings
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train a Naive Bayes model
    nb_model = MultinomialNB()
    nb_model.fit(X_train_vec, y_train)
    
    # Evaluate the model
    y_pred = nb_model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return nb_model, vectorizer, accuracy, report, X_test, y_test, y_pred

# Predict disease based on symptoms
def predict_disease(model, vectorizer, symptoms_input, data):
    # Convert symptoms input to the same format as training data
    symptoms_str = ','.join(symptoms_input)
    symptoms_vec = vectorizer.transform([symptoms_str])
    
    # Get prediction probabilities
    proba = model.predict_proba(symptoms_vec)[0]
    
    # Get top 5 diseases with highest probabilities
    top_indices = proba.argsort()[-5:][::-1]
    top_diseases = [model.classes_[i] for i in top_indices]
    top_probabilities = [proba[i] for i in top_indices]
    
    # Get associated information for each disease
    results = []
    for disease, probability in zip(top_diseases, top_probabilities):
        disease_info = data[data['disease'] == disease]
        
        if not disease_info.empty:
            risk_level = disease_info['risk level'].values[0] if 'risk level' in disease_info.columns else "Unknown"
            doctor = disease_info['doctor'].values[0] if 'doctor' in disease_info.columns else "Unknown"
            cures = disease_info['cures'].values[0] if 'cures' in disease_info.columns else "Unknown"
            
            results.append({
                'disease': disease,
                'probability': probability * 100,  # Convert to percentage
                'risk_level': risk_level,
                'doctor': doctor,
                'cures': cures
            })
    
    return results

# Streamlit UI
def main():
    st.set_page_config(page_title="Disease Prediction System", layout="wide")
    
    st.title("Human Disease Prediction Based on Symptoms")
    st.write("Select symptoms to predict possible diseases and their accuracy levels")
    
    # Load and prepare data
    data = load_data()
    disease_symptom_dict, unique_symptoms = preprocess_data(data)
    
    # Build model
    model, vectorizer, accuracy, report, X_test, y_test, y_pred = build_model(data, unique_symptoms)
    
    # Sidebar for model metrics
    st.sidebar.header("Model Performance")
    st.sidebar.write(f"Model Accuracy: {accuracy:.2f}")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Disease Prediction", "Model Analysis", "Dataset Overview"])
    
    with tab1:
        st.header("Select Symptoms")
        
        # Create columns for better layout
        col1, col2, col3 = st.columns(3)
        
        # Split symptoms into three columns for better UI
        symptoms_per_column = len(unique_symptoms) // 3
        
        selected_symptoms = []
        
        with col1:
            for symptom in unique_symptoms[:symptoms_per_column]:
                if st.checkbox(symptom):
                    selected_symptoms.append(symptom)
        
        with col2:
            for symptom in unique_symptoms[symptoms_per_column:2*symptoms_per_column]:
                if st.checkbox(symptom):
                    selected_symptoms.append(symptom)
        
        with col3:
            for symptom in unique_symptoms[2*symptoms_per_column:]:
                if st.checkbox(symptom):
                    selected_symptoms.append(symptom)
        
        if st.button("Predict Disease"):
            if len(selected_symptoms) > 0:
                results = predict_disease(model, vectorizer, selected_symptoms, data)
                
                st.header("Prediction Results")
                
                for result in results:
                    with st.expander(f"{result['disease']} - Confidence: {result['probability']:.2f}%"):
                        st.write(f"**Risk Level:** {result['risk_level']}")
                        st.write(f"**Recommended Doctor:** {result['doctor']}")
                        st.write(f"**Possible Treatments:** {result['cures']}")
                
                # Visualization of prediction probabilities
                fig, ax = plt.subplots(figsize=(10, 6))
                diseases = [r['disease'] for r in results]
                probabilities = [r['probability'] for r in results]
                
                bars = ax.barh(diseases, probabilities, color='skyblue')
                ax.set_xlabel('Confidence (%)')
                ax.set_ylabel('Disease')
                ax.set_title('Disease Prediction Confidence Levels')
                
                # Add percentage labels on bars
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                            ha='left', va='center')
                
                st.pyplot(fig)
            else:
                st.warning("Please select at least one symptom.")
    
    with tab2:
        st.header("Model Analysis")
        
        # Confusion Matrix Visualization
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
        
        # Use a subset for better visualization if there are many classes
        if len(model.classes_) > 10:
            # Find the most frequent classes in the test set
            class_counts = pd.Series(y_test).value_counts().head(10).index.tolist()
            indices = [list(model.classes_).index(cls) for cls in class_counts]
            cm_subset = cm[indices, :][:, indices]
            classes_subset = [model.classes_[i] for i in indices]
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm_subset, annot=True, fmt='d', cmap='Blues', xticklabels=classes_subset, yticklabels=classes_subset)
        else:
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
        
        plt.ylabel('True Disease')
        plt.xlabel('Predicted Disease')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        st.pyplot(fig)
        
        # Feature importance analysis
        st.subheader("Symptom Importance")
        feature_names = vectorizer.get_feature_names_out()
        
        # Get feature importance from model coefficients
        feature_importance = np.mean(model.feature_log_prob_, axis=0)
        
        # Get top 15 features
        top_indices = np.argsort(feature_importance)[-15:]
        top_features = [feature_names[i] for i in top_indices]
        top_importance = feature_importance[top_indices]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(top_features, top_importance, color='lightgreen')
        ax.set_xlabel('Importance Score')
        ax.set_ylabel('Symptoms')
        ax.set_title('Top 15 Important Symptoms in Disease Prediction')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Classification Report
        st.subheader("Classification Report")
        st.text(report)
    
    with tab3:
        st.header("Dataset Overview")
        
        # Display sample of dataset
        st.subheader("Sample Data")
        st.dataframe(data.head())
        
        # Disease distribution
        st.subheader("Disease Distribution")
        fig, ax = plt.subplots(figsize=(12, 6))
        data['disease'].value_counts().head(20).plot(kind='bar', ax=ax)
        plt.title('Top 20 Diseases in Dataset')
        plt.ylabel('Count')
        plt.xticks(rotation=90)
        st.pyplot(fig)
        
        # Risk level distribution
        st.subheader("Risk Level Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        risk_counts = data['risk level'].value_counts()
        ax.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        plt.title('Distribution of Risk Levels')
        st.pyplot(fig)

if __name__ == "__main__":
    main()
