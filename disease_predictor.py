import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

class DiseasePredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Human Disease Prediction System")
        self.root.geometry("800x600")
        
        # Load and preprocess data
        self.df = pd.read_csv("dataset.csv")
        self.preprocess_data()
        self.train_model()
        
        # GUI Components
        self.create_widgets()
    
    def preprocess_data(self):
        # Clean data
        self.df = self.df.drop_duplicates()
        
        # Extract all unique symptoms
        all_symptoms = set()
        for symptoms in self.df['symptoms']:
            for symptom in symptoms.split(','):
                all_symptoms.add(symptom.strip().lower())
        
        # Create binary features for each symptom
        for symptom in all_symptoms:
            self.df[symptom] = self.df['symptoms'].apply(lambda x: 1 if symptom in x.lower() else 0)
        
        # Encode target variable
        self.le = LabelEncoder()
        self.df['disease_encoded'] = self.le.fit_transform(self.df['disease'])
        
        # Prepare features and target
        self.X = self.df.drop(['disease', 'symptoms', 'cures', 'doctor', 'risk level', 'disease_encoded'], axis=1)
        self.y = self.df['disease_encoded']
    
    def train_model(self):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Train Random Forest
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Calculate accuracy
        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
    
    def create_widgets(self):
        # Title
        title_label = ttk.Label(
            self.root, 
            text="Disease Prediction System", 
            font=("Helvetica", 16, "bold")
        )
        title_label.pack(pady=10)
        
        # Accuracy display
        acc_label = ttk.Label(
            self.root,
            text=f"Model Accuracy: {self.accuracy:.2%}",
            font=("Helvetica", 12)
        )
        acc_label.pack(pady=5)
        
        # Symptoms selection frame
        symptoms_frame = ttk.LabelFrame(
            self.root, 
            text="Select Symptoms", 
            padding=(10, 5)
        )
        symptoms_frame.pack(pady=10, padx=20, fill="both", expand=True)
        
        # Create checkboxes for symptoms
        self.symptom_vars = {}
        
        # Get top 50 most common symptoms for the GUI (to keep it manageable)
        symptom_counts = {}
        for symptoms in self.df['symptoms']:
            for symptom in symptoms.split(','):
                s = symptom.strip().lower()
                symptom_counts[s] = symptom_counts.get(s, 0) + 1
        
        top_symptoms = sorted(symptom_counts.items(), key=lambda x: x[1], reverse=True)[:50]
        
        # Arrange checkboxes in 3 columns
        cols = 3
        rows = (len(top_symptoms) // cols) + 1
        
        for i, (symptom, count) in enumerate(top_symptoms):
            var = tk.IntVar()
            self.symptom_vars[symptom] = var
            
            row = i // cols
            col = i % cols
            
            cb = ttk.Checkbutton(
                symptoms_frame,
                text=f"{symptom.title()}",
                variable=var,
                onvalue=1,
                offvalue=0
            )
            cb.grid(row=row, column=col, sticky="w", padx=5, pady=2)
        
        # Prediction button
        predict_btn = ttk.Button(
            self.root,
            text="Predict Disease",
            command=self.predict_disease
        )
        predict_btn.pack(pady=10)
        
        # Result frame
        self.result_frame = ttk.LabelFrame(
            self.root, 
            text="Prediction Result", 
            padding=(10, 5)
        )
        self.result_frame.pack(pady=10, padx=20, fill="both", expand=True)
        
        # Result labels (initially empty)
        self.disease_label = ttk.Label(self.result_frame, text="", font=("Helvetica", 12, "bold"))
        self.disease_label.pack(pady=5, anchor="w")
        
        self.symptoms_label = ttk.Label(self.result_frame, text="", wraplength=700)
        self.symptoms_label.pack(pady=5, anchor="w")
        
        self.cures_label = ttk.Label(self.result_frame, text="", wraplength=700)
        self.cures_label.pack(pady=5, anchor="w")
        
        self.doctor_label = ttk.Label(self.result_frame, text="", wraplength=700)
        self.doctor_label.pack(pady=5, anchor="w")
        
        self.risk_label = ttk.Label(self.result_frame, text="", wraplength=700)
        self.risk_label.pack(pady=5, anchor="w")
    
    def predict_disease(self):
        # Create input vector based on selected symptoms
        input_data = [0] * len(self.X.columns)
        
        for symptom, var in self.symptom_vars.items():
            if symptom in self.X.columns and var.get() == 1:
                idx = list(self.X.columns).index(symptom)
                input_data[idx] = 1
        
        # Make prediction
        try:
            prediction = self.model.predict([input_data])
            disease = self.le.inverse_transform(prediction)[0]
            
            # Get disease details from dataset
            disease_info = self.df[self.df['disease'] == disease].iloc[0]
            
            # Update result labels
            self.disease_label.config(text=f"Predicted Disease: {disease}")
            self.symptoms_label.config(text=f"Symptoms: {disease_info['symptoms']}")
            self.cures_label.config(text=f"Treatment: {disease_info['cures']}")
            self.doctor_label.config(text=f"Recommended Doctor: {disease_info['doctor']}")
            self.risk_label.config(text=f"Risk Level: {disease_info['risk level']}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DiseasePredictorApp(root)
    root.mainloop()
