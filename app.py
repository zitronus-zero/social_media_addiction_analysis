import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Page Configuration
st.set_page_config(
    page_title="Social Media Addiction & Mental Wellbeing Predictor",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS Styling    
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #4f46e5;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #4338ca;
        color: white;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Title and Introduction
st.title("🧠 Social Media Addiction & Mental Wellbeing Predictor")
st.markdown("Aplikasi berbasis Machine Learning untuk menganalisis dan memprediksi tingkat kecanduan media sosial serta dampaknya terhadap kesejahteraan mental.")

@st.cache_resource
def train_model():
    # Membuat synthetic dataset yang representatif untuk demo pelatihan model
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Age': np.random.randint(15, 50, n_samples),
        'Daily_Usage_Hours': np.random.uniform(1.0, 12.0, n_samples),
        'Posts_Per_Day': np.random.randint(0, 15, n_samples),
        'Anxiety_Score': np.random.randint(1, 10, n_samples),
        'Depression_Score': np.random.randint(1, 10, n_samples),
        'Sleep_Quality_Hours': np.random.uniform(3.0, 9.0, n_samples),
        'Loneliness_Score': np.random.randint(1, 10, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Feature Engineering
    df['Engagement_Ratio'] = df['Posts_Per_Day'] / (df['Daily_Usage_Hours'] + 0.1)
    df['Total_Distress'] = df['Anxiety_Score'] + df['Depression_Score'] + df['Loneliness_Score']
    
    # Target rule simulation for training
    conditions = [
        (df['Daily_Usage_Hours'] > 8) & (df['Total_Distress'] > 20),
        (df['Daily_Usage_Hours'] > 5) & (df['Total_Distress'] > 12),
        (df['Daily_Usage_Hours'] > 25)
    ]
    # Addiction Levels: 0: Low, 1: Moderate, 2: High, 3: Severe
    def assign_level(row):
        score = row['Daily_Usage_Hours'] * 1.5 + row['Total_Distress'] * 0.8 - row['Sleep_Quality_Hours'] * 1.2
        if score < 10:
            return 'Low'
        elif score < 18:
            return 'Moderate'
        elif score < 26:
            return 'High'
        else:
            return 'Severe'
            
    df['Addiction_Level'] = df.apply(assign_level, axis=1)
    
    X = df.drop(columns=['Addiction_Level'])
    y = df['Addiction_Level']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y_encoded)
    
    return model, scaler, le, X.columns.tolist()

model, scaler, label_encoder, feature_columns = train_model()

# Sidebar Inputs
st.sidebar.header("📝 Masukkan Parameter Pengguna")
age = st.sidebar.slider("Usia (Tahun)", 15, 60, 20)
daily_hours = st.sidebar.slider("Durasi Penggunaan Harian (Jam)", 0.5, 16.0, 4.5, 0.5)
posts_per_day = st.sidebar.slider("Jumlah Unggahan / Post per Hari", 0, 30, 3)
anxiety = st.sidebar.slider("Skor Kecemasan (1-10)", 1, 10, 4)
depression = st.sidebar.slider("Skor Depresi (1-10)", 1, 10, 3)
sleep_hours = st.sidebar.slider("Durasi / Kualitas Tidur (Jam)", 2.0, 10.0, 7.0, 0.5)
loneliness = st.sidebar.slider("Skor Kesunyian (1-10)", 1, 10, 4)

# Main Dashboard Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Ringkasan Data Input")
    input_data = {
        'Age': [age],
        'Daily_Usage_Hours': [daily_hours],
        'Posts_Per_Day': [posts_per_day],
        'Anxiety_Score': [anxiety],
        'Depression_Score': [depression],
        'Sleep_Quality_Hours': [sleep_hours],
        'Loneliness_Score': [loneliness]
    }
    df_input = pd.DataFrame(input_data)
    
    # Calculate engineered features
    df_input['Engagement_Ratio'] = df_input['Posts_Per_Day'] / (df_input['Daily_Usage_Hours'] + 0.1)
    df_input['Total_Distress'] = df_input['Anxiety_Score'] + df_input['Depression_Score'] + df_input['Loneliness_Score']
    
    # Reorder columns to match training
    df_input = df_input[feature_columns]
    
    st.dataframe(df_input.T, use_container_width=True)

with col2:
    st.subheader("🔮 Hasil Prediksi Model")
    if st.button("Jalankan Prediksi Kecanduan"):
        input_scaled = scaler.transform(df_input)
        prediction_encoded = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]
        
        # Color mapping for prediction
        color_map = {'Low': 'green', 'Moderate': 'blue', 'High': 'orange', 'Severe': 'red'}
        
        st.markdown(f"### Tingkat Kecanduan: **{prediction_label}**")
        
        # Display probabilities
        st.markdown("#### Probabilitas Kelas:")
        prob_df = pd.DataFrame({
            'Tingkat': label_encoder.classes_,
            'Probabilitas': prediction_proba
        })
        st.bar_chart(prob_df.set_index('Tingkat'))
        
        # Recommendations
        st.markdown("#### 💡 Rekomendasi Kesejahteraan Mental:")
        if prediction_label in ['High', 'Severe']:
            st.warning("⚠️ Penggunaan media sosial Anda cukup tinggi dan terindikasi memberikan dampak psikologis. Pertimbangkan untuk melakukan *digital detox*, membatasi notifikasi, serta memperbanyak aktivitas offline.")
        elif prediction_label == 'Moderate':
            st.info("ℹ️ Pola penggunaan cukup seimbang, namun tetap perhatikan kualitas tidur dan tingkat stres harian Anda.")
        else:
            st.success("✨ Bagus! Kebiasaan digital Anda berada dalam batas yang sehat dan terkontrol.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Dikembangkan untuk Analisis Kecanduan Media Sosial & Kesejahteraan Mental | Streamlit Deployment</p>", unsafe_allow_html=True)
