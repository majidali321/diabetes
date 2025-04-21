import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Diabetes Prediction App", layout="wide")

# Function: Load and preprocess data
@st.cache_data
def load_and_preprocess_data():
    try:
        df1 = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv', low_memory=False)
        df2 = pd.read_csv('diabetes_binary_5050split_health_indicators_BRFSS2015.csv', low_memory=False)
        df3 = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv', low_memory=False)

        valid_dfs = [df for df in [df1, df2, df3] if 'Diabetes_binary' in df.columns]
        df = pd.concat(valid_dfs, ignore_index=True)

        df['Diabetes_binary'] = pd.to_numeric(df['Diabetes_binary'], errors='coerce')
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=['Diabetes_binary'])
        df['Diabetes_binary'] = df['Diabetes_binary'].astype(int)

        if 'BMI' not in df.columns:
            df['BMI'] = np.nan
        df['BMI'] = df['BMI'].fillna(df['BMI'].median())

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.remove('Diabetes_binary')

        imputer = SimpleImputer(strategy='median')
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

        return df
    except Exception as e:
        st.error(f"Data loading error: {e}")
        return None

# Function: EDA
def eda(df):
    st.subheader("Exploratory Data Analysis (EDA)")
    st.write(df.describe())
    st.write("Target Distribution")
    st.bar_chart(df['Diabetes_binary'].value_counts())

    corr = df.corr()
    important = corr.index[abs(corr['Diabetes_binary']) > 0.1]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr.loc[important, important], annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Function: Train models
def train_models(X_train, y_train):
    models = {}

    rf_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(random_state=42))
    ])
    rf_params = {
        'rf__max_depth': [5, 10],
        'rf__n_estimators': [100],
    }
    grid_rf = GridSearchCV(rf_pipe, rf_params, cv=3, scoring='roc_auc', n_jobs=-1)
    grid_rf.fit(X_train, y_train)
    models['Random Forest'] = grid_rf

    svm_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, random_state=42))
    ])
    svm_params = {
        'svm__C': [0.1, 1],
        'svm__gamma': ['scale'],
        'svm__kernel': ['rbf']
    }
    grid_svm = GridSearchCV(svm_pipe, svm_params, cv=3, scoring='roc_auc', n_jobs=-1)
    grid_svm.fit(X_train, y_train)
    models['SVM'] = grid_svm

    return models

# Function: Evaluate models
def evaluate(models, X_test, y_test):
    st.subheader("Model Evaluation")
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        st.markdown(f"### {name}")
        st.write("Best Params:", model.best_params_)
        st.write(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
        st.write(f"Precision: {precision_score(y_test, y_pred):.4f}")
        st.write(f"Recall: {recall_score(y_test, y_pred):.4f}")
        st.write(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

# ================================
# Main Streamlit UI
# ================================
st.title("🩺 Diabetes Risk Prediction System")

# Load data
df = load_and_preprocess_data()

if df is not None:

    # Sidebar: Prediction Interface
    st.sidebar.header("🔮 Prediction Interface")

    # Initialize session state for models and data features
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = None
    if 'X_columns' not in st.session_state and 'Diabetes_binary' in df.columns:
        st.session_state.X_columns = df.drop(columns=['Diabetes_binary']).columns

    # Sidebar input form for prediction
    with st.sidebar.form("prediction_form"):
        st.subheader("Patient Parameters")

        col1, col2 = st.columns(2)

        with col1:
            HighBP = st.selectbox("High BP", [0, 1], index=0)
            HighChol = st.selectbox("High Cholesterol", [0, 1], index=0)
            BMI = st.slider("BMI", 10.0, 50.0, 25.0)
            Smoker = st.selectbox("Smoker", [0, 1], index=0)
            Stroke = st.selectbox("Stroke History", [0, 1], index=0)
            HeartDisease = st.selectbox("Heart Disease", [0, 1], index=0)
            PhysActivity = st.selectbox("Physical Activity", [0, 1], index=1)

        with col2:
            Fruits = st.selectbox("Fruit Consumption", [0, 1], index=1)
            Veggies = st.selectbox("Vegetable Consumption", [0, 1], index=1)
            Alcohol = st.selectbox("Heavy Alcohol", [0, 1], index=0)
            Healthcare = st.selectbox("Healthcare Access", [0, 1], index=1)
            GenHlth = st.slider("General Health (1-5)", 1, 5, 3)
            MentHlth = st.slider("Mental Health Days", 0, 30, 3)
            PhysHlth = st.slider("Physical Health Days", 0, 30, 3)

        DiffWalk = st.selectbox("Difficulty Walking", [0, 1], index=0)
        Sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        Age = st.slider("Age Category", 1, 13, 5)
        Education = st.slider("Education Level", 1, 6, 4)
        Income = st.slider("Income Level", 1, 8, 6)

        submitted = st.form_submit_button("Predict Risk")

    # Show raw data if requested
    if st.checkbox("Show raw data"):
        st.dataframe(df.head(100))

    # Run EDA if requested
    if st.checkbox("Run EDA"):
        eda(df)

    # Train models button and logic
    if st.button("Train Models"):
        with st.spinner("Training in progress..."):
            X = df.drop(columns=['Diabetes_binary'])
            y = df['Diabetes_binary']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            st.session_state.trained_models = train_models(X_train, y_train)
            st.session_state.X_columns = X.columns  # Save feature columns
            evaluate(st.session_state.trained_models, X_test, y_test)

    # Prediction handling
    if submitted:
        if st.session_state.trained_models is None:
            st.warning("Please train the models first by clicking the 'Train Models' button.")
        else:
            input_features = pd.DataFrame([{
                'HighBP': HighBP,
                'HighChol': HighChol,
                'BMI': BMI,
                'Smoker': Smoker,
                'Stroke': Stroke,
                'HeartDiseaseorAttack': HeartDisease,
                'PhysActivity': PhysActivity,
                'Fruits': Fruits,
                'Veggies': Veggies,
                'HvyAlcoholConsump': Alcohol,
                'AnyHealthcare': Healthcare,
                'GenHlth': GenHlth,
                'MentHlth': MentHlth,
                'PhysHlth': PhysHlth,
                'DiffWalk': DiffWalk,
                'Sex': Sex,
                'Age': Age,
                'Education': Education,
                'Income': Income
            }])

            # Ensure input features have the same order and columns as training data
            input_features = input_features.reindex(columns=st.session_state.X_columns, fill_value=0)

            st.subheader("📊 Prediction Results")
            for name, model in st.session_state.trained_models.items():
                proba = model.predict_proba(input_features)[0][1]
                st.metric(
                    label=f"{name} Risk Probability",
                    value=f"{proba:.1%}",
                    help=f"Probability of having diabetes according to {name}"
                )
else:
    st.error("Failed to load data. Please check your CSV files and try again.")
