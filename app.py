import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

# --- Page Configuration ---
st.set_page_config(
    page_title="Medical Cost Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
# Using a slightly updated, more modern color palette and typography
st.markdown("""
<style>
    /* Main background and text color */
    body {
        background-color: #f0f2f6;
        color: #333333;
    }

    /* Streamlit's main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    /* Headers */
    .main-header {
        font-size: 3.2rem; /* Larger for main title */
        color: #004d99; /* Darker blue for prominence */
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 700;
        letter-spacing: -0.05rem;
    }
    .sub-header {
        font-size: 1.8rem; /* Slightly larger for section headers */
        color: #0066cc; /* Mid blue */
        margin-bottom: 1.2rem;
        font-weight: 600;
        border-bottom: 2px solid #e0e0e0; /* Subtle divider */
        padding-bottom: 0.5rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #0066cc;
        border-bottom: 2px solid #e0e0e0;
        margin-top: 2.5rem;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        font-weight: 600;
    }
    
    /* Info text box */
    .info-text {
        font-size: 1.1rem;
        background-color: #e6f7ff; /* Lighter blue background */
        color: #003366; /* Dark blue text */
        padding: 15px 20px;
        border-left: 5px solid #007bff; /* Accent border */
        border-radius: 8px;
        margin-bottom: 25px;
        line-height: 1.6;
    }

    /* Sidebar styles */
    .stSidebar {
        background-color: #ffffff; /* White background */
        border-right: 1px solid #e0e0e0;
        padding-top: 2rem;
    }
    .stSidebar .stRadio > label {
        font-size: 1.1rem;
        padding: 8px 0;
    }
    .stSidebar .stRadio > label:hover {
        background-color: #f0f2f6; /* Light hover effect */
        border-radius: 5px;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background-color: #f7f9fc;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px 20px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #666666;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #0066cc;
        font-weight: 700;
    }
    [data-testid="stMetricDelta"] svg {
        display: none; /* Hide delta arrows for cleaner look on non-delta metrics */
    }

    /* Forms and buttons */
    .stButton > button {
        background-color: #007bff;
        color: white;
        border-radius: 5px;
        padding: 0.6rem 1.2rem;
        font-size: 1.1rem;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #0056b3;
    }
    .st-dg { /* DataFrame */
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f0f2f6;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        padding: 10px 15px;
        font-size: 1.1rem;
        font-weight: 600;
        color: #0066cc;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size:1.1rem;
        color: #004d99;
    }
    .stTabs [data-baseweb="tab-list"] button {
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        margin-right: 5px;
        padding: 10px 20px;
        border: 1px solid #e0e0e0;
        border-bottom: none;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 3px solid #007bff;
        font-weight: 600;
        color: #007bff;
    }

    /* General text improvements */
    p, li {
        font-size: 1.05rem;
        line-height: 1.7;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<h1 class='main-header'>🏥 Medical Insurance Cost Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='info-text'>Analyze and predict medical insurance costs based on personal factors using advanced machine learning models.</p>", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("health.png", width=90) # Slightly smaller icon
    st.markdown("<h2 class='sub-header'>Navigation</h2>", unsafe_allow_html=True)
    page = st.radio(
        "Explore the app:",
        ["🏠 Home", "📊 Data Exploration", "📈 Prediction", "📋 Model Info"],
        key="main_navigation_radio", # Added a key for clarity
    )
    st.markdown("---") # Visual separator
    st.markdown(
        """
        <div style="font-size: 0.9rem; color: #777;">
            Developed by Ankit<br>
            Data Source: Kaggle
        </div>
        """, unsafe_allow_html=True
    )


# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv('insurance.csv')
    return df

df = load_data()

# --- Train/Load Model ---
# No functional changes here, only added docstrings for clarity
def train_model():
    """Trains the models and saves the artifacts."""
    df_ml = df.copy()
    
    # Encode categorical variables
    le_sex = LabelEncoder()
    le_smoker = LabelEncoder()
    le_region = LabelEncoder()
    
    df_ml['sex_encoded'] = le_sex.fit_transform(df_ml['sex'])
    df_ml['smoker_encoded'] = le_smoker.fit_transform(df_ml['smoker'])
    df_ml['region_encoded'] = le_region.fit_transform(df_ml['region'])
    
    # Create new features
    df_ml['age_squared'] = df_ml['age'] ** 2
    df_ml['bmi_squared'] = df_ml['bmi'] ** 2
    df_ml['age_bmi_interaction'] = df_ml['age'] * df_ml['bmi']
    df_ml['smoker_age_interaction'] = df_ml['smoker_encoded'] * df_ml['age']
    df_ml['smoker_bmi_interaction'] = df_ml['smoker_encoded'] * df_ml['bmi']
    
    # BMI categories
    df_ml['bmi_category'] = pd.cut(df_ml['bmi'],
                                   bins=[0, 18.5, 25, 30, float('inf')],
                                   labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    df_ml['is_obese'] = (df_ml['bmi'] >= 30).astype(int)
    
    # Age categories
    df_ml['age_category'] = pd.cut(df_ml['age'],
                                   bins=[0, 25, 35, 45, 55, float('inf')],
                                   labels=['Young', 'Young_Adult', 'Middle_Age', 'Pre_Senior', 'Senior'])
    
    # One-hot encode new categorical features
    bmi_dummies = pd.get_dummies(df_ml['bmi_category'], prefix='bmi', dtype=int) # Added dtype=int for clarity
    age_dummies = pd.get_dummies(df_ml['age_category'], prefix='age', dtype=int) # Added dtype=int
    df_ml = pd.concat([df_ml, bmi_dummies, age_dummies], axis=1)
    
    # Select features for modeling
    base_features = ['age', 'sex_encoded', 'bmi', 'children', 'smoker_encoded', 'region_encoded']
    engineered_features = ['age_squared', 'bmi_squared', 'age_bmi_interaction',
                           'smoker_age_interaction', 'smoker_bmi_interaction', 'is_obese']
    categorical_features = list(bmi_dummies.columns) + list(age_dummies.columns)
    
    all_features = base_features + engineered_features + categorical_features
    X = df_ml[all_features]
    y = df_ml['charges']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train Ridge Regression model
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    
    # Train Random Forest model
    rf_model = RandomForestRegressor(
        n_estimators=300, 
        max_depth=10, 
        min_samples_leaf=4, 
        min_samples_split=10, 
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    
    # Save the model artifacts
    if not os.path.exists('models'):
        os.makedirs('models')
        
    model_artifacts = {
        'ridge_model': ridge_model,
        'rf_model': rf_model,
        'label_encoders': {
            'sex': le_sex,
            'smoker': le_smoker,
            'region': le_region
        },
        'feature_names': all_features,
        'bmi_dummies_columns': list(bmi_dummies.columns),
        'age_dummies_columns': list(age_dummies.columns)
    }
    
    with open('models/insurance_prediction_model.pkl', 'wb') as f:
        pickle.dump(model_artifacts, f)
    
    return model_artifacts

@st.cache_resource # Using st.cache_resource for models/artifacts
def get_model():
    """Loads pre-trained models or trains them if they don't exist."""
    if not os.path.exists('models/insurance_prediction_model.pkl'):
        st.info("Training models for the first time... This might take a moment.")
        model_artifacts = train_model()
        st.success("Models trained and saved successfully!")
    else:
        with open('models/insurance_prediction_model.pkl', 'rb') as f:
            model_artifacts = pickle.load(f)
    return model_artifacts

model_artifacts = get_model()

# --- Predict Function ---
# No functional changes here, only added docstrings and ensured dummy handling
def predict_insurance_cost(age, sex, bmi, children, smoker, region, model_type='ridge'):
    """Predicts insurance cost based on input features."""
    # Create input data
    person_data = {
        'age': age, 'sex': sex, 'bmi': bmi,
        'children': children, 'smoker': smoker, 'region': region
    }
    
    # Get encoders and features
    label_encoders = model_artifacts['label_encoders']
    feature_names = model_artifacts['feature_names']
    bmi_dummies_columns = model_artifacts['bmi_dummies_columns']
    age_dummies_columns = model_artifacts['age_dummies_columns']
    
    # Create DataFrame
    person_df = pd.DataFrame([person_data])
    
    # Apply same preprocessing
    person_df['sex_encoded'] = label_encoders['sex'].transform(person_df['sex'])
    person_df['smoker_encoded'] = label_encoders['smoker'].transform(person_df['smoker'])
    person_df['region_encoded'] = label_encoders['region'].transform(person_df['region'])
    
    # Create engineered features
    person_df['age_squared'] = person_df['age'] ** 2
    person_df['bmi_squared'] = person_df['bmi'] ** 2
    person_df['age_bmi_interaction'] = person_df['age'] * person_df['bmi']
    person_df['smoker_age_interaction'] = person_df['smoker_encoded'] * person_df['age']
    person_df['smoker_bmi_interaction'] = person_df['smoker_encoded'] * person_df['bmi']
    person_df['is_obese'] = (person_df['bmi'] >= 30).astype(int)
    
    # BMI and age categories
    person_df['bmi_category'] = pd.cut(person_df['bmi'],
                                        bins=[0, 18.5, 25, 30, float('inf')],
                                        labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    person_df['age_category'] = pd.cut(person_df['age'],
                                        bins=[0, 25, 35, 45, 55, float('inf')],
                                        labels=['Young', 'Young_Adult', 'Middle_Age', 'Pre_Senior', 'Senior'])
    
    # Initialize all dummy columns to 0
    for col in bmi_dummies_columns:
        person_df[col] = 0
    for col in age_dummies_columns:
        person_df[col] = 0
    
    # Set appropriate dummy variables for the single input row
    if not pd.isna(person_df['bmi_category'].iloc[0]):
        bmi_cat = person_df['bmi_category'].iloc[0]
        bmi_col = f"bmi_{bmi_cat}"
        if bmi_col in person_df.columns:
            person_df[bmi_col] = 1
    
    if not pd.isna(person_df['age_category'].iloc[0]):
        age_cat = person_df['age_category'].iloc[0]
        age_col = f"age_{age_cat}"
        if age_col in person_df.columns:
            person_df[age_col] = 1

    # Ensure all feature_names are present, even if some are all zeros for this input
    # Create a DataFrame with all expected columns, then fill with input data
    X_processed = pd.DataFrame(columns=feature_names)
    X_processed.loc[0] = 0 # Initialize a row with zeros
    for col in person_df.columns:
        if col in X_processed.columns:
            X_processed[col] = person_df[col]
            
    # Select features and predict
    X_new = X_processed[feature_names] # Use the fully prepared dataframe

    if model_type == 'ridge':
        prediction = model_artifacts['ridge_model'].predict(X_new)[0]
    else:
        prediction = model_artifacts['rf_model'].predict(X_new)[0]
    
    return prediction

# --- Home page ---
def home_page():
    st.markdown("<h2 class='section-header'>Welcome to the Medical Insurance Cost Predictor</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        This application helps analyze and predict **medical insurance costs** based on various personal factors like age, Body Mass Index (BMI), smoking status, and more. Understanding these costs is crucial for financial planning and health awareness.
        
        ### Key Features:
        - **Data Exploration**: Dive deep into the dataset with interactive charts and statistics.
        - **Cost Prediction**: Get a personalized estimate of your annual insurance premium.
        - **Model Information**: Learn about the machine learning algorithms powering the predictions and their performance.
        
        ### About the Dataset:
        The dataset contains anonymized information about medical insurance costs for individuals, capturing key demographic and health indicators:
        - **Age**: Primary beneficiary's age
        - **Sex**: Gender of the primary beneficiary
        - **BMI (Body Mass Index)**: Body Mass Index, providing an understanding of body, relative to height and weight, objectively.
        - **Children**: Number of children/dependents covered by health insurance
        - **Smoker**: Whether the beneficiary smokes or not
        - **Region**: The beneficiary’s residential area in the US (northeast, northwest, southeast, southwest)
        - **Charges**: Individual medical costs billed by health insurance (the target variable)
        """)
    
    with col2:
        st.markdown("<h3 style='text-align: center; color: #0066cc;'>Quick Dataset Stats</h3>", unsafe_allow_html=True)
        # Using columns for better alignment of metrics
        metrics_col1, metrics_col2 = st.columns(2)
        with metrics_col1:
            st.metric("Total Records", f"{df.shape[0]:,}")
            st.metric("Average BMI", f"{df['bmi'].mean():.1f}")
            st.metric("Average Insurance Cost", f"${df['charges'].mean():,.2f}")
        with metrics_col2:
            st.metric("Average Age", f"{df['age'].mean():.1f} years")
            st.metric("Smokers in Data", f"{(df['smoker'] == 'yes').sum():,} ({(df['smoker'] == 'yes').mean()*100:.1f}%)")
            st.metric("Non-Smokers in Data", f"{(df['smoker'] == 'no').sum():,} ({(df['smoker'] == 'no').mean()*100:.1f}%)")
        
    st.markdown("---") # Visual separator
    st.markdown("<h3 class='sub-header'>Sample Data</h3>", unsafe_allow_html=True)
    st.dataframe(df.sample(5, random_state=42), use_container_width=True) # Added random_state for consistent sample

    st.markdown("---") # Visual separator
    st.markdown("<h3 class='sub-header'>Getting Started</h3>", unsafe_allow_html=True)
    st.markdown("""
    1. Navigate to the **📊 Data Exploration** page to see detailed analysis and trends within the dataset.
    2. Try the **📈 Prediction** page to estimate insurance costs based on your personal information.
    3. Check the **📋 Model Information** page to learn more about the machine learning models and their performance.
    """)

# --- Data exploration page ---
def exploration_page():
    st.markdown("<h2 class='section-header'>📊 Data Exploration & Visualization</h2>", unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset Overview", "📈 Distributions", "🔄 Correlations", "💰 Cost Analysis"])
    
    with tab1:
        st.markdown("<h3 style='color: #0066cc;'>Dataset Information</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Dataset Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
            st.info(f"**Total Missing Values:** {df.isnull().sum().sum()}")
        
        with col2:
            st.info(f"**Categorical Features:** {', '.join(df.select_dtypes(include=['object']).columns.tolist())}")
            st.info(f"**Numerical Features:** {', '.join(df.select_dtypes(include=['number']).columns.tolist())}")
        
        st.markdown("<h3 style='color: #0066cc;'>Statistical Summary</h3>", unsafe_allow_html=True)
        st.dataframe(df.describe().T, use_container_width=True) # Transpose for better readability
        
        st.markdown("<h3 style='color: #0066cc;'>Categorical Value Counts</h3>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Sex Distribution")
            st.dataframe(df['sex'].value_counts())
        
        with col2:
            st.subheader("Smoker Status")
            st.dataframe(df['smoker'].value_counts())
        
        with col3:
            st.subheader("Region Distribution")
            st.dataframe(df['region'].value_counts())
    
    with tab2:
        st.markdown("<h3 style='color: #0066cc;'>Feature Distributions</h3>", unsafe_allow_html=True)
        
        # Age distribution
        st.subheader("Age Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['age'], bins=20, kde=True, ax=ax, color='#007bff')
        ax.axvline(df['age'].mean(), color='red', linestyle='--', label=f'Mean: {df["age"].mean():.1f}')
        ax.axvline(df['age'].median(), color='green', linestyle='--', label=f'Median: {df["age"].median():.1f}')
        ax.legend()
        ax.set_title('Age Distribution', fontsize=16)
        ax.set_xlabel('Age', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        st.pyplot(fig)
        plt.close(fig) # Close plot to prevent memory issues

        col1, col2 = st.columns(2)
        
        with col1:
            # BMI distribution
            st.subheader("BMI Distribution")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(df['bmi'], bins=20, kde=True, ax=ax, color='#28a745')
            ax.axvline(df['bmi'].mean(), color='red', linestyle='--', label=f'Mean: {df["bmi"].mean():.1f}')
            ax.axvline(df['bmi'].median(), color='green', linestyle='--', label=f'Median: {df["bmi"].median():.1f}')
            ax.axvline(30, color='orange', linestyle='--', label='Obesity Line (30)')
            ax.legend()
            ax.set_title('BMI Distribution', fontsize=16)
            ax.set_xlabel('BMI', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            st.pyplot(fig)
            plt.close(fig)
        
        with col2:
            # Children distribution
            st.subheader("Number of Children")
            fig, ax = plt.subplots(figsize=(8, 6))
            # FIX: Add hue and legend=False to address FutureWarning
            sns.countplot(x='children', data=df, ax=ax, palette='viridis', hue='children', legend=False)
            ax.set_title('Children Count', fontsize=16)
            ax.set_xlabel('Number of Children', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            st.pyplot(fig)
            plt.close(fig)
        
        # Charges distribution
        st.subheader("Insurance Charges Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['charges'], bins=30, kde=True, ax=ax, color='#ffc107')
        ax.axvline(df['charges'].mean(), color='red', linestyle='--', label=f'Mean: ${df["charges"].mean():,.1f}')
        ax.axvline(df['charges'].median(), color='green', linestyle='--', label=f'Median: ${df["charges"].median():,.1f}')
        ax.legend()
        ax.set_title('Insurance Charges Distribution', fontsize=16)
        ax.set_xlabel('Charges ($)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        st.pyplot(fig)
        plt.close(fig)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Sex distribution
            st.subheader("Sex Distribution")
            fig, ax = plt.subplots(figsize=(6, 6))
            df['sex'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, ax=ax, colors=['#6a0572', '#ab83a7']) # Custom colors
            ax.set_title('Sex Distribution', fontsize=14)
            ax.set_ylabel('') # Hide default ylabel
            st.pyplot(fig)
            plt.close(fig)
        
        with col2:
            # Smoker distribution
            st.subheader("Smoker Status")
            fig, ax = plt.subplots(figsize=(6, 6))
            df['smoker'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, ax=ax, colors=['#dc3545', '#28a745']) # Red for smoker, green for non-smoker
            ax.set_title('Smoker Status', fontsize=14)
            ax.set_ylabel('')
            st.pyplot(fig)
            plt.close(fig)
        
        with col3:
            # Region distribution
            st.subheader("Region Distribution")
            fig, ax = plt.subplots(figsize=(6, 6))
            df['region'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, ax=ax, cmap='Pastel1') # Color palette
            ax.set_title('Region Distribution', fontsize=14)
            ax.set_ylabel('')
            st.pyplot(fig)
            plt.close(fig)
    
    with tab3:
        st.markdown("<h3 style='color: #0066cc;'>Feature Correlations</h3>", unsafe_allow_html=True)
        
        numerical_cols = ['age', 'bmi', 'children', 'charges']
        correlation_matrix = df[numerical_cols].corr()
        
        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.3f', linewidths=0.5, ax=ax, cbar_kws={'label': 'Correlation Coefficient'})
        ax.set_title('Correlation Matrix of Numerical Features', fontsize=16)
        st.pyplot(fig)
        plt.close(fig)
        
        # Scatter plots
        st.subheader("Scatter Plots")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### Age vs. Charges")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(x='age', y='charges', data=df, hue='smoker', palette={'yes': '#dc3545', 'no': '#28a745'}, ax=ax, s=60, alpha=0.7) # Added size and alpha
            ax.set_title('Age vs. Charges (colored by smoker status)', fontsize=16)
            ax.set_xlabel('Age', fontsize=12)
            ax.set_ylabel('Charges ($)', fontsize=12)
            ax.legend(title='Smoker')
            st.pyplot(fig)
            plt.close(fig)
        
        with col2:
            st.write("#### BMI vs. Charges")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(x='bmi', y='charges', data=df, hue='smoker', palette={'yes': '#dc3545', 'no': '#28a745'}, ax=ax, s=60, alpha=0.7)
            ax.set_title('BMI vs. Charges (colored by smoker status)', fontsize=16)
            ax.set_xlabel('BMI', fontsize=12)
            ax.set_ylabel('Charges ($)', fontsize=12)
            ax.legend(title='Smoker')
            st.pyplot(fig)
            plt.close(fig)
    
    with tab4:
        st.markdown("<h3 style='color: #0066cc;'>Insurance Cost Analysis</h3>", unsafe_allow_html=True)
        
        # Charges by smoker status
        st.subheader("Insurance Charges by Smoker Status")
        fig, ax = plt.subplots(figsize=(10, 6))
        # FIX: Add hue and legend=False to address FutureWarning
        sns.boxplot(x='smoker', y='charges', data=df, ax=ax, palette={'yes': '#dc3545', 'no': '#28a745'}, hue='smoker', legend=False)
        ax.set_title('Insurance Charges by Smoker Status', fontsize=16)
        ax.set_xlabel('Smoker', fontsize=12)
        ax.set_ylabel('Charges ($)', fontsize=12)
        st.pyplot(fig)
        plt.close(fig)
        
        # Calculate average charges by smoker status
        smoker_charges = df[df['smoker'] == 'yes']['charges'].mean()
        non_smoker_charges = df[df['smoker'] == 'no']['charges'].mean()
        
        cols = st.columns(2)
        with cols[0]:
            st.metric("Smoker Average Charges", f"${smoker_charges:,.2f}")
        with cols[1]:
            st.metric("Non-Smoker Average Charges", f"${non_smoker_charges:,.2f}")
        
        st.markdown(f"""
        <p style='font-size:1.1rem; line-height:1.6;'>
        On average, **smokers pay approximately <span style='color:#dc3545; font-weight:bold;'>{smoker_charges/non_smoker_charges:.1f}x</span> more** than non-smokers for medical insurance. This highlights smoking as a dominant factor in cost determination.
        </p>
        """, unsafe_allow_html=True)
        
        # Charges by region
        st.subheader("Insurance Charges by Region")
        fig, ax = plt.subplots(figsize=(10, 6))
        # FIX: Add hue and legend=False to address FutureWarning
        sns.boxplot(x='region', y='charges', data=df, ax=ax, palette='viridis', hue='region', legend=False)
        ax.set_title('Insurance Charges by Region', fontsize=16)
        ax.set_xlabel('Region', fontsize=12)
        ax.set_ylabel('Charges ($)', fontsize=12)
        st.pyplot(fig)
        plt.close(fig)
        
        # Average charges by region
        region_charges = df.groupby('region')['charges'].mean().sort_values(ascending=False)
        
        st.subheader("Average Charges by Region")
        
        cols = st.columns(4)
        for i, (region, charge) in enumerate(region_charges.items()):
            with cols[i]:
                st.metric(f"{region.capitalize()}", f"${charge:,.2f}")
        
        # Charges by age groups
        st.subheader("Charges by Age Groups")
        df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 100], labels=['18-25', '26-35', '36-45', '46-55', '56+'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        # FIX: Add hue and legend=False to address FutureWarning
        sns.boxplot(x='age_group', y='charges', data=df, ax=ax, palette='coolwarm', hue='age_group', legend=False)
        ax.set_title('Insurance Charges by Age Group', fontsize=16)
        ax.set_xlabel('Age Group')
        ax.set_ylabel('Charges ($)')
        st.pyplot(fig)
        plt.close(fig)

# --- Prediction page ---
def prediction_page():
    st.markdown("<h2 class='section-header'>📈 Insurance Cost Prediction</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        <p class='info-text'>
            Input your personal details below to get an estimated annual medical insurance cost.
            You can choose between two machine learning models for the prediction:
            <b>Ridge Regression</b> (a simpler, robust linear model) or <b>Random Forest</b> (a powerful, non-linear ensemble model).
        </p>
        """, unsafe_allow_html=True)
        
        with st.form("prediction_form", clear_on_submit=False): # Added clear_on_submit=False
            st.markdown("<h3>Enter Your Information</h3>", unsafe_allow_html=True)
            
            form_col1, form_col2 = st.columns(2)
            
            with form_col1:
                age = st.slider("Age", min_value=18, max_value=64, value=30, help="Your age in years")
                sex = st.selectbox("Sex", options=["male", "female"], help="Your biological sex", index=0)
                bmi = st.slider("BMI", min_value=15.0, max_value=53.0, value=25.0, step=0.1, 
                                 help="Body Mass Index (weight in kg / height in m²)")
            
            with form_col2:
                children = st.slider("Number of Children/Dependents", min_value=0, max_value=5, value=0, help="Number of dependents covered by insurance")
                smoker = st.selectbox("Smoker", options=["no", "yes"], help="Are you a smoker?", index=0)
                region = st.selectbox("Region", 
                                     options=["northeast", "northwest", "southeast", "southwest"],
                                     help="Your residential region in the US", index=2) # Default to southeast for commonality
            
            st.markdown("---") # Separator for model selection
            model_type = st.radio("Select Prediction Model", ["Ridge Regression", "Random Forest"], 
                                  help="Ridge is simpler; Random Forest often captures more complex patterns.", horizontal=True) # Horizontal radio buttons
            
            submitted = st.form_submit_button("💰 Predict Insurance Cost")
        
        if submitted:
            # Make prediction
            with st.spinner("Calculating your estimated insurance cost..."):
                model = 'ridge' if model_type == "Ridge Regression" else 'rf'
                predicted_cost = predict_insurance_cost(age, sex, bmi, children, smoker, region, model)
            
            # FIX: Removed unsafe_allow_html from st.success
            st.success(f"### Estimated Annual Insurance Cost: ${predicted_cost:,.2f}") 
            # If you still want the blue color, you can use st.markdown separately:
            st.markdown(f"<h3 style='color:#007bff;'>Predicted Cost: ${predicted_cost:,.2f}</h3>", unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("<h3>Factors Influencing Your Prediction:</h3>", unsafe_allow_html=True)
            factors = []
            
            if smoker == "yes":
                factors.append("🚨 **Smoking status:** This is the single most significant factor, dramatically increasing costs.")
            
            if bmi >= 30:
                factors.append("⚠️ **BMI in the obese range (≥30):** This typically leads to higher insurance premiums due to associated health risks.")
            elif bmi >= 25:
                factors.append("🔶 **BMI in the overweight range (25-29.9):** This may also contribute to increased costs.")
            else:
                factors.append("✅ **Healthy BMI (18.5-24.9):** Your BMI is within the normal range, a positive factor for costs.")

            if age >= 55:
                factors.append("⬆️ **Age 55 or older:** Insurance costs generally increase with age, reflecting higher health risks.")
            elif age >= 40:
                factors.append("📈 **Age 40-54:** Costs may start to increase as you enter middle age.")
            else:
                factors.append("🟢 **Younger Age:** Your age is in a lower-risk category, which helps keep costs down.")
            
            if children >= 3:
                factors.append("👨‍👩‍👧‍👦 **Three or more children/dependents:** More dependents can slightly increase your insurance costs.")
            
            if region == "southeast":
                factors.append("📍 **Southeast region:** Historically, this region has slightly higher average medical costs.")
            
            if not factors:
                st.info("✅ Based on your inputs, your profile has relatively few high-risk factors for insurance costs.")
            else:
                for factor in factors:
                    st.markdown(factor)
            
    with col2:
        st.markdown("<h3 style='color: #0066cc;'>Understanding the Factors</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        #### How medical insurance costs are typically determined:
        
        - **Smoking**: The most impactful factor. Smokers often face significantly higher premiums due to increased health risks.
        - **Age**: Premiums generally rise with age, as health care needs tend to increase over time.
        - **BMI**: Higher Body Mass Index, particularly in the obese range, can lead to increased costs due to potential health complications.
        - **Region**: Geographical location can influence costs due to variations in local healthcare prices, regulations, and competition.
        - **Number of Children/Dependents**: Having more dependents covered by your policy can modestly increase the overall cost.
        - **Sex**: Generally, gender has a minimal direct impact on insurance costs, as per many regulations.
        """)
        
        st.markdown("---") # Visual separator
        st.markdown("<h3 style='color: #0066cc;'>BMI Categories</h3>", unsafe_allow_html=True)
        st.markdown("""
        - **Underweight**: BMI below 18.5
        - **Normal weight**: BMI between 18.5 and 24.9
        - **Overweight**: BMI between 25 and 29.9
        - **Obese**: BMI 30 or higher
        """)
        
        st.markdown("---") # Visual separator
        st.markdown("<h3 style='color: #0066cc;'>Reference Cost Points</h3>", unsafe_allow_html=True)
        
        # Calculate averages for reference
        avg_smoker = df[df['smoker'] == 'yes']['charges'].mean()
        avg_nonsmoker = df[df['smoker'] == 'no']['charges'].mean()
        avg_obese = df[df['bmi'] >= 30]['charges'].mean()
        avg_normal_bmi = df[(df['bmi'] >= 18.5) & (df['bmi'] < 25)]['charges'].mean()
        
        st.markdown(f"""
        - Average cost for **Smokers**: **${avg_smoker:,.2f}**
        - Average cost for **Non-Smokers**: **${avg_nonsmoker:,.2f}**
        - Average cost for **Obese individuals**: **${avg_obese:,.2f}**
        - Average cost for **Normal BMI individuals**: **${avg_normal_bmi:,.2f}**
        - **Overall average cost**: **${df['charges'].mean():,.2f}**
        """)

# --- Model info page ---
def model_info_page():
    st.markdown("<h2 class='section-header'>📋 Model Information</h2>", unsafe_allow_html=True)
    
    st.markdown("<p class='info-text'>Explore the machine learning models used for insurance cost prediction, their performance, and key insights.</p>", unsafe_allow_html=True)
    
    st.markdown("### Model Performance Overview")
    st.markdown("""
    We rigorously trained and evaluated several machine learning models on the medical insurance dataset. Below is a summary of their performance on a held-out test set:
    
    | Model | R² Score (Test) | RMSE (Test) | MAE (Test) | Cross-Validation R² Score |
    | :-------------------- | :-------------: | :----------: | :----------: | :-----------------------: |
    | **Ridge Regression** | **0.8497** | **4,597.65** | **2,708.53** | **0.8442** |
    | **Random Forest** | **0.8497** | **4,597.42** | **2,426.33** | **0.8394** |
    | Linear Regression       | 0.8490          | 4,609.40     | 2,728.71     | 0.8441                    |
    | Gradient Boosting       | 0.8474          | 4,633.29     | 2,463.84     | 0.8510                    |
    | Lasso Regression        | 0.8491          | 4,606.98     | 2,723.60     | 0.8441                    |
    | Support Vector Regression | 0.7127        | 6,357.65     | 2,755.18     | 0.7048                    |
    
    *Note: RMSE and MAE are in USD ($). Lower values indicate better performance.*
    """)
    
    st.markdown("---")
    st.markdown("### Key Feature Importances")
    
    st.markdown("""
    Understanding which factors contribute most to insurance costs is crucial. Based on our Random Forest model, here are the top influencing features:
    """)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Feature importance plot
        st.subheader("Random Forest Feature Importance")
        
        # Ensure 'smoker_encoded' is converted to 'Smoker (Encoded)' for display if it's high
        # Or even better, map directly to 'Smoker'
        # For this example, I'll update the labels to be more readable for the user.
        feature_labels = {
            'smoker_bmi_interaction': 'Smoker x BMI Interaction',
            'age': 'Age',
            'age_squared': 'Age Squared', 
            'age_bmi_interaction': 'Age x BMI Interaction', 
            'smoker_age_interaction': 'Smoker x Age Interaction',
            'children': 'Number of Children', 
            'smoker_encoded': 'Smoker Status', 
            'bmi_squared': 'BMI Squared', 
            'bmi': 'BMI', 
            'region_encoded': 'Region'
        }
        
        # Assuming these were derived from a trained Random Forest model's .feature_importances_
        top_features_raw = [
            'smoker_bmi_interaction', 'age', 'age_squared', 
            'age_bmi_interaction', 'smoker_age_interaction',
            'children', 'smoker_encoded', 'bmi_squared', 
            'bmi', 'region_encoded'
        ]
        importance_values = [
            0.7923, 0.0517, 0.0473, 0.0343, 0.0230,
            0.0118, 0.0101, 0.0082, 0.0076, 0.0064
        ]
        
        # Map raw feature names to display labels
        top_features_display = [feature_labels.get(f, f) for f in top_features_raw]

        fig, ax = plt.subplots(figsize=(10, 8))
        y_pos = np.arange(len(top_features_display))
        
        ax.barh(y_pos, importance_values, color='#007bff') # Consistent color
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features_display, fontsize=12)
        ax.invert_yaxis() # Highest importance at the top
        ax.set_xlabel('Feature Importance (Relative Score)', fontsize=12)
        ax.set_title('Top 10 Feature Importance by Random Forest', fontsize=16)
        st.pyplot(fig)
        plt.close(fig)
        
    with col2:
        st.markdown("### Model Details")
        
        st.markdown("""
        **Ridge Regression (L2 Regularization)**
        - **Type:** Linear Regression variant
        - **Purpose:** Addresses multicollinearity and prevents overfitting by adding a penalty equivalent to the square of the magnitude of coefficients.
        - **Strengths:** Robust, stable, and works well when features are correlated.
        
        **Random Forest Regressor**
        - **Type:** Ensemble Learning (Bagging)
        - **Purpose:** Builds multiple decision trees and merges their predictions to get a more accurate and stable prediction.
        - **Strengths:** Highly accurate, can handle non-linear relationships, less prone to overfitting than single decision trees, and provides direct feature importance.
        """)
        
        st.markdown("### Feature Engineering Explained")
        st.markdown("""
        To improve the models' ability to capture complex patterns in medical costs, we created several new features from the raw data:
        
        - **Age Squared & BMI Squared:** Account for non-linear relationships, as the impact of age or BMI on costs might accelerate at higher values.
        - **Age-BMI Interaction:** Captures how the combined effect of age and BMI influences charges (e.g., being older and having a high BMI).
        - **Smoker-Age Interaction & Smoker-BMI Interaction:** Essential features that highlight how smoking status heavily modulates the impact of age and BMI on costs. For instance, a high BMI might be more costly for a smoker than a non-smoker.
        - **BMI Categories (e.g., 'Obese'):** Converts continuous BMI into discrete groups (Underweight, Normal, Overweight, Obese) which can capture step-changes in cost risk.
        - **Age Categories (e.g., 'Senior'):** Similar to BMI categories, grouping ages can help the model identify cost patterns related to life stages.
        """)
    
    st.markdown("---")
    st.markdown("### Understanding Evaluation Metrics")
    
    col_metrics_1, col_metrics_2 = st.columns(2)
    
    with col_metrics_1:
        st.markdown("""
        **R² Score (Coefficient of Determination)**
        - **What it is:** Measures the proportion of the variance in the dependent variable (charges) that is predictable from the independent variables.
        - **Interpretation:** A value closer to 1.0 indicates that the model explains a large portion of the variance in the target variable.
        """)
        st.markdown("""
        **Mean Absolute Error (MAE)**
        - **What it is:** The average of the absolute differences between actual and predicted values.
        - **Interpretation:** Represents the average magnitude of the errors in a set of predictions, without considering their direction. It's in the same unit as the target variable ($).
        """)
    
    with col_metrics_2:
        st.markdown("""
        **Root Mean Squared Error (RMSE)**
        - **What it is:** The square root of the average of the squared differences between actual and predicted values.
        - **Interpretation:** Gives a relatively high weight to large errors. It's also in the same unit as the target variable ($). Generally, a lower RMSE is better.
        """)
        st.markdown("""
        **Cross-Validation R² Score (CV R²)**
        - **What it is:** An R² score derived from K-fold cross-validation, where the data is split into 'K' subsets, and the model is trained and tested K times.
        - **Interpretation:** Provides a more robust estimate of the model's performance on unseen data by reducing reliance on a single train-test split.
        """)

# --- Main App Routing ---
if page == "🏠 Home":
    home_page()
elif page == "📊 Data Exploration":
    exploration_page()
elif page == "📈 Prediction":
    prediction_page()
elif page == "📋 Model Info":
    model_info_page()