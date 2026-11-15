import streamlit as st
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import os

st.set_page_config(page_title="Depression Risk Predictor", layout="wide")
st.title("🧠 Mental Health & Depression Risk Predictor")

# 1. Custom Transformer (Motha Logic)
class CustomDataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.financial_stress_median_ = None
        self.rare_jobs_ = None
        self.train_columns_ = None
        
    def fit(self, X, y=None):
        self.financial_stress_median_ = X['Financial Stress'].median()
        job_counts = X['Profession'].value_counts()
        self.rare_jobs_ = job_counts[job_counts < 10].index.tolist()
        X_transformed = self._apply_transform(X, is_fit=True)
        self.train_columns_ = X_transformed.columns.tolist()
        return self
        
    def transform(self, X):
        return self._apply_transform(X, is_fit=False)
        
    def _apply_transform(self, X, is_fit=False):
        df = X.copy()
        cols_to_drop = ['id', 'Name', 'City', 'Degree', 'Depression']
        df = df.drop([c for c in cols_to_drop if c in df.columns], axis=1, errors='ignore')
        
        gender_map = {'Female': 0, 'Male': 1}
        yes_no_map = {'No': 0, 'Yes': 1}
        work_stu_map = {'Student': 0, 'Working Professional': 1}
        
        df['Gender'] = df['Gender'].map(gender_map)
        df['Have you ever had suicidal thoughts ?'] = df['Have you ever had suicidal thoughts ?'].map(yes_no_map)
        df['Family History of Mental Illness'] = df['Family History of Mental Illness'].map(yes_no_map)
        df['Working Professional or Student'] = df['Working Professional or Student'].map(work_stu_map)
        
        valid_sleep = ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours']
        df['Sleep Duration'] = df['Sleep Duration'].apply(lambda x: x if x in valid_sleep else '5-6 hours')
        sleep_map = {'Less than 5 hours': 0, '5-6 hours': 1, '7-8 hours': 2, 'More than 8 hours': 3}
        df['Sleep Duration'] = df['Sleep Duration'].map(sleep_map)
        
        valid_diet = ['Moderate', 'Unhealthy', 'Healthy']
        df['Dietary Habits'] = df['Dietary Habits'].apply(lambda x: x if x in valid_diet else 'Moderate')
        diet_map = {'Unhealthy': 0, 'Moderate': 1, 'Healthy': 2}
        df['Dietary Habits'] = df['Dietary Habits'].map(diet_map)
        
        df['Financial Stress'] = df['Financial Stress'].fillna(self.financial_stress_median_)
        
        zero_cols = ['Academic Pressure', 'Study Satisfaction', 'CGPA', 'Work Pressure', 'Job Satisfaction']
        for col in zero_cols:
            df[col] = df[col].fillna(0.0)
            
        df['Profession'] = df['Profession'].fillna('Unemployed')
        df['Profession'] = df['Profession'].replace({'Finanancial Analyst': 'Financial Analyst'})
        df['Profession'] = df['Profession'].replace(self.rare_jobs_, 'Unemployed')
        
        df = pd.get_dummies(df, columns=['Profession'], drop_first=True, dtype=int)
        
        if not is_fit:
            df = df.reindex(columns=self.train_columns_, fill_value=0)
            
        return df

# 2. PyTorch Architecture
class DepressionMLP(nn.Module):
    def __init__(self, input_dim):
        super(DepressionMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

# 3. Load Models (The Bulletproof Method)
@st.cache_resource
def load_models():
    # We rebuild the pipeline instead of relying on the pickle file
    pipeline = Pipeline([
        ('cleaner', CustomDataPreprocessor()),
        ('scaler', StandardScaler())
    ])
    
    # Check if train.csv exists in the current directory
    if not os.path.exists('train.csv'):
         st.error("Error: 'train.csv' not found. Please place it in the same folder as this script to initialize the model.")
         st.stop()
         
    df_raw = pd.read_csv('train.csv')
    X_train = df_raw.drop('Depression', axis=1, errors='ignore')
    
    # Fit the pipeline
    pipeline.fit(X_train)
    
    # Get the exact output dimension dynamically
    dummy_processed = pipeline.transform(X_train.head(1))
    input_dim = dummy_processed.shape[1]
    
    model = DepressionMLP(input_dim=input_dim)
    model.load_state_dict(torch.load('depression_model.pth', weights_only=True))
    model.eval()
    
    return pipeline, model

pipeline, model = load_models()

# ==========================================
# UI DESIGN: 2 TABS (Single vs Bulk)
# ==========================================
tab1, tab2 = st.tabs(["👤 Single Profile Prediction", "📂 Bulk Prediction (CSV Upload)"])

# ----------------- TAB 1 -----------------
with tab1:
    with st.form("user_input_form"):
        st.subheader("Your Profile Details")
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Age", min_value=15, max_value=80, value=25)
            work_student = st.selectbox("Working Professional or Student", ["Working Professional", "Student"])
            profession = st.text_input("Profession (e.g., Teacher, Engineer)", value="Unemployed")
        with col2:
            sleep = st.selectbox("Sleep Duration", ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"])
            diet = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])
            suicidal = st.selectbox("Have you ever had suicidal thoughts ?", ["No", "Yes"])
            family_hist = st.selectbox("Family History of Mental Illness", ["No", "Yes"])

        st.subheader("Pressure & Satisfaction (Scale 1.0 to 5.0)")
        c1, c2, c3 = st.columns(3)
        with c1:
            academic_press = st.number_input("Academic Pressure", min_value=0.0, max_value=5.0, value=0.0)
            work_press = st.number_input("Work Pressure", min_value=0.0, max_value=5.0, value=0.0)
        with c2:
            cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=0.0)
            work_study_hours = st.number_input("Work/Study Hours", min_value=0.0, max_value=24.0, value=8.0)
        with c3:
            study_sat = st.number_input("Study Satisfaction", min_value=0.0, max_value=5.0, value=0.0)
            job_sat = st.number_input("Job Satisfaction", min_value=0.0, max_value=5.0, value=0.0)
            financial_stress = st.number_input("Financial Stress", min_value=1.0, max_value=5.0, value=3.0)

        submit_button = st.form_submit_button("Predict Risk")

    if submit_button:
        user_data = pd.DataFrame([{
            'Gender': gender, 'Age': age, 'Working Professional or Student': work_student,
            'Profession': profession, 'Academic Pressure': academic_press, 'Work Pressure': work_press,
            'CGPA': cgpa, 'Study Satisfaction': study_sat, 'Job Satisfaction': job_sat,
            'Sleep Duration': sleep, 'Dietary Habits': diet, 'Have you ever had suicidal thoughts ?': suicidal,
            'Work/Study Hours': work_study_hours, 'Financial Stress': financial_stress,
            'Family History of Mental Illness': family_hist
        }])

        processed_data = pipeline.transform(user_data)
        tensor_data = torch.tensor(processed_data, dtype=torch.float32)
        with torch.no_grad():
            risk_prob = model(tensor_data).item()
        
        st.markdown("---")
        if risk_prob >= 0.5:
            st.error(f"⚠️ High Risk of Depression Detected (Probability: {risk_prob*100:.1f}%)")
        else:
            st.success(f"✅ Low Risk of Depression (Probability: {risk_prob*100:.1f}%)")

# ----------------- TAB 2 -----------------
with tab2:
    st.subheader("Upload Employee/User Data (CSV)")
    
    st.info("""
    **Required Columns in CSV:**
    `Gender`, `Age`, `Working Professional or Student`, `Profession`, `Academic Pressure`, 
    `Work Pressure`, `CGPA`, `Study Satisfaction`, `Job Satisfaction`, `Sleep Duration`, 
    `Dietary Habits`, `Have you ever had suicidal thoughts ?`, `Work/Study Hours`, 
    `Financial Stress`, `Family History of Mental Illness`
    """)

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df_bulk = pd.read_csv(uploaded_file)
            st.write(f"✅ Loaded {len(df_bulk)} rows successfully.")
            
            processed_bulk = pipeline.transform(df_bulk)
            
            tensor_bulk = torch.tensor(processed_bulk, dtype=torch.float32)
            with torch.no_grad():
                probs_bulk = model(tensor_bulk).squeeze().numpy()
            
            df_bulk['Risk_Probability (%)'] = (probs_bulk * 100).round(2)
            df_bulk['Depression_Prediction'] = ['Yes' if p >= 0.5 else 'No' for p in probs_bulk]
            
            st.dataframe(df_bulk[['Risk_Probability (%)', 'Depression_Prediction'] + df_bulk.columns[:-2].tolist()])
            
            csv = df_bulk.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predicted Results",
                data=csv,
                file_name='depression_predictions.csv',
                mime='text/csv',
            )
        except Exception as e:
            st.error(f"Error processing file. Please check column names. Error: {e}")