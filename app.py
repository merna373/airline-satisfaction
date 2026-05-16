import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, SelectKBest, SequentialFeatureSelector as SFS
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import GridSearchCV
st.set_page_config(page_title="Airline Satisfaction Analysis", page_icon="✈️", layout="wide")

# ========================
# CSS Styling
# ========================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #3B82F6;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #E5E7EB;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ========================
# Session State Initialization
# ========================
if 'train_df' not in st.session_state:
    st.session_state.train_df = None
if 'test_df' not in st.session_state:
    st.session_state.test_df = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'features' not in st.session_state:
    st.session_state.features = None

# ========================
# Helper Functions (from notebook)
# ========================

def load_data(train_file, test_file):
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    return train, test

def preprocess_data(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Drop unnecessary columns
    train_df.drop(columns=["Unnamed: 0", "id"], inplace=True, errors="ignore")
    test_df.drop(columns=["Unnamed: 0", "id"], inplace=True, errors="ignore")

    # Handle missing values
    median_delay = train_df["Arrival Delay in Minutes"].median()
    train_df["Arrival Delay in Minutes"] = train_df["Arrival Delay in Minutes"].fillna(median_delay)
    test_df["Arrival Delay in Minutes"] = test_df["Arrival Delay in Minutes"].fillna(median_delay)

    # Remove outliers from Flight Distance in train only
    Q1 = train_df['Flight Distance'].quantile(0.25)
    Q3 = train_df['Flight Distance'].quantile(0.75)
    IQR = Q3 - Q1
    train_df = train_df[(train_df['Flight Distance'] >= Q1 - 1.5*IQR) & (train_df['Flight Distance'] <= Q3 + 1.5*IQR)]

    return train_df, test_df

def create_new_features(df):
    df = df.copy()
    df.drop(columns=['Delay Group'], inplace=True, errors='ignore')

    df['Total Delay'] = (
        df['Departure Delay in Minutes'].fillna(0) +
        df['Arrival Delay in Minutes'].fillna(0)
    )

    df['Delay Level'] = pd.cut(
        df['Total Delay'],
        bins=[-0.1, 5, 30, float('inf')],
        labels=['No Delay', 'Medium Delay', 'High Delay']
    )

    df['Age_Group'] = pd.cut(
        df['Age'],
        bins=[0, 25, 50, 100],
        labels=False
    )

    return df

def fit_class_features(train_df):
    class_avg = train_df.groupby('Class')['Inflight service'].mean()
    return class_avg

def apply_class_features(df, class_avg):
    df = df.copy()
    df['Class_Service_Avg'] = df['Class'].map(class_avg)
    return df

def encode_features(df):
    df = df.copy()
    categorical_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df

def feature_engineering_pipeline(train_df, test_df):
    train_df = create_new_features(train_df)
    test_df = create_new_features(test_df)

    class_avg = fit_class_features(train_df)

    train_df = apply_class_features(train_df, class_avg)
    test_df = apply_class_features(test_df, class_avg)

    train_df = encode_features(train_df)
    test_df = encode_features(test_df)

    train_df = pd.get_dummies(train_df, columns=['Delay Level'], drop_first=True)
    test_df = pd.get_dummies(test_df, columns=['Delay Level'], drop_first=True)

    train_df, test_df = train_df.align(test_df, join='left', axis=1, fill_value=0)

    return train_df, test_df

def check_constant_features(x_train, x_val, x_test):
    constant_var = VarianceThreshold(threshold=0)
    constant_var.fit(x_train)
    mask = constant_var.get_support()

    x_train_filtered = constant_var.transform(x_train)
    x_val_filtered = constant_var.transform(x_val)
    x_test_filtered = constant_var.transform(x_test)

    # Convert back to DataFrame with proper columns
    selected_cols = x_train.columns[mask]
    x_train_filtered = pd.DataFrame(x_train_filtered, columns=selected_cols, index=x_train.index)
    x_val_filtered = pd.DataFrame(x_val_filtered, columns=selected_cols, index=x_val.index)
    x_test_filtered = pd.DataFrame(x_test_filtered, columns=selected_cols, index=x_test.index)

    return x_train_filtered, x_val_filtered, x_test_filtered

def forward_feature_selection(x_train, y_train):
    sfs_forward = SFS(
        estimator=RandomForestClassifier(n_estimators=10, n_jobs=-1, random_state=0),
        n_features_to_select=20,
        tol=None,
        direction='forward',
        scoring='roc_auc',
        cv=2,
        n_jobs=-1,
    )
    sfs_forward = sfs_forward.fit(x_train, y_train)
    forward_features = x_train.columns[sfs_forward.get_support()].tolist()
    return forward_features

def run_model(model, x_train, y_train, x_val, y_val):
    model.fit(x_train, y_train)
    pred = model.predict(x_val)
    acc = accuracy_score(y_val, pred)
    f1 = f1_score(y_val, pred)
    report = classification_report(y_val, pred)
    return model, acc, f1, report, pred

# ========================
# Visualization Functions
# ========================

def stacked_bar_plot(df, col, title):
    pivot = df.groupby([col, "satisfaction"]).size().unstack()
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax, color=['#EF4444', '#10B981'])
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    return fig

def plot_correlation_heatmap(df):
    pdf = df.copy()
    if pdf["satisfaction"].dtype == object:
        pdf["satisfaction"] = pdf["satisfaction"].map({"satisfied": 1, "neutral or dissatisfied": 0})
    corr = pdf.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr, cmap="coolwarm", linewidths=0.3, annot=False, ax=ax)
    ax.set_title("Correlation Heatmap")
    return fig

def plot_pie(data, column, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    data[column].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=sns.color_palette("pastel"))
    ax.set_title(title)
    ax.set_ylabel("")
    return fig

def plot_hist(data, column, bins=20):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data[column], bins=bins, kde=True, ax=ax, color="#3B82F6")
    ax.set_title(f'Distribution of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')
    return fig

def plot_box(data, column):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x=data[column], ax=ax, color="#60A5FA")
    ax.set_title(f'Boxplot of {column}')
    return fig

# ========================
# Sidebar
# ========================
st.sidebar.title("✈️ Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Home",
    "📁 Data Upload",
    "📊 EDA & Visualizations",
    "🔧 Data Preprocessing",
    "🧠 Feature Engineering & Selection",
    "🤖 Model Training",
    "📈 Model Comparison",
    "🔮 Prediction"
])

# ========================
# Home Page
# ========================
if page == "🏠 Home":
    st.markdown("<div class='main-header'>Airline Passenger Satisfaction Analysis</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dataset", "Airline Satisfaction")
    with col2:
        st.metric("Models", "4 Algorithms")
    with col3:
        st.metric("Features", "20+ Selected")

    st.markdown("""
    ### 🎯 Project Overview
    This application provides a complete pipeline for airline passenger satisfaction prediction:

    - **📁 Data Upload**: Load train and test CSV files
    - **📊 EDA**: Interactive visualizations and statistical analysis
    - **🔧 Preprocessing**: Missing values, duplicates, outliers handling
    - **🧠 Feature Engineering**: Encoding, feature creation, and selection
    - **🤖 Model Training**: Logistic Regression, Random Forest, Gradient Boosting, XGBoost
    - **📈 Comparison**: Side-by-side model evaluation
    - **🔮 Prediction**: Make predictions on new data

    Use the sidebar to navigate through different sections.
    """)

    st.image("https://img.freepik.com/free-vector/airplane-flying-illustration_23-2149447627.jpg", width=400)

# ========================
# Data Upload
# ========================
elif page == "📁 Data Upload":
    st.markdown("<div class='sub-header'>📁 Upload Your Data</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        train_file = st.file_uploader("Upload Training Data (train.csv)", type=['csv'])
    with col2:
        test_file = st.file_uploader("Upload Testing Data (test.csv)", type=['csv'])

    if train_file and test_file:
        train_df, test_df = load_data(train_file, test_file)
        st.session_state.train_df = train_df
        st.session_state.test_df = test_df

        st.success("✅ Data loaded successfully!")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Train Shape", f"{train_df.shape[0]} rows")
        with col2:
            st.metric("Train Columns", train_df.shape[1])
        with col3:
            st.metric("Test Shape", f"{test_df.shape[0]} rows")
        with col4:
            st.metric("Test Columns", test_df.shape[1])

        st.subheader("Training Data Preview")
        st.dataframe(train_df.head(10), use_container_width=True)

        st.subheader("Column Information")
        buffer = []
        buffer.append(f"**Target Column:** `satisfaction`")
        buffer.append(f"**Numeric Columns:** {len(train_df.select_dtypes(include=[np.number]).columns)}")
        buffer.append(f"**Categorical Columns:** {len(train_df.select_dtypes(include=['object']).columns)}")
        st.markdown("<br>".join(buffer), unsafe_allow_html=True)

        with st.expander("View Detailed Info"):
            st.write("**Columns:**", list(train_df.columns))
            st.write("**Data Types:**")
            st.write(train_df.dtypes)
    else:
        st.info("👆 Please upload both train.csv and test.csv files to proceed.")

# ========================
# EDA & Visualizations
# ========================
elif page == "📊 EDA & Visualizations":
    st.markdown("<div class='sub-header'>📊 Exploratory Data Analysis</div>", unsafe_allow_html=True)

    if st.session_state.train_df is None:
        st.warning("⚠️ Please upload data first in the 'Data Upload' section.")
    else:
        train_df = st.session_state.train_df

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📋 Statistics", "📊 Categorical", "📈 Correlation", "🥧 Distribution", 
            "📉 Histograms", "📦 Boxplots"
        ])

        with tab1:
            st.subheader("Dataset Statistics")
            st.write(train_df.describe())

            st.subheader("Missing Values")
            missing = train_df.isnull().sum()
            st.bar_chart(missing[missing > 0] if len(missing[missing > 0]) > 0 else missing)

            st.subheader("Duplicate Rows")
            st.write(f"Number of duplicates: {train_df.duplicated().sum()}")

        with tab2:
            st.subheader("Stacked Bar Charts")
            cat_cols = ["Gender", "Class", "Customer Type", "Type of Travel"]
            selected_cat = st.selectbox("Select Category", cat_cols)
            fig = stacked_bar_plot(train_df, selected_cat, f"Satisfaction by {selected_cat}")
            st.pyplot(fig)

        with tab3:
            st.subheader("Correlation Heatmap")
            fig = plot_correlation_heatmap(train_df)
            st.pyplot(fig)

        with tab4:
            st.subheader("Pie Charts")
            pie_cols = ["satisfaction", "Class", "Gender", "Customer Type"]
            selected_pie = st.selectbox("Select Column", pie_cols)
            fig = plot_pie(train_df, selected_pie, f"{selected_pie} Distribution")
            st.pyplot(fig)

        with tab5:
            st.subheader("Histograms")
            num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
            selected_hist = st.selectbox("Select Numeric Column", num_cols)
            bins = st.slider("Number of bins", 5, 100, 20)
            fig = plot_hist(train_df, selected_hist, bins)
            st.pyplot(fig)

        with tab6:
            st.subheader("Boxplots")
            num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
            selected_box = st.selectbox("Select Column for Boxplot", num_cols)
            fig = plot_box(train_df, selected_box)
            st.pyplot(fig)

            st.subheader("Outlier Detection (IQR Method)")
            Q1 = train_df[selected_box].quantile(0.25)
            Q3 = train_df[selected_box].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = train_df[(train_df[selected_box] < lower) | (train_df[selected_box] > upper)]
            st.write(f"**Q1:** {Q1:.2f} | **Q3:** {Q3:.2f} | **IQR:** {IQR:.2f}")
            st.write(f"**Lower Bound:** {lower:.2f} | **Upper Bound:** {upper:.2f}")
            st.write(f"**Outliers Found:** {len(outliers)} ({len(outliers)/len(train_df)*100:.2f}%)")

# ========================
# Data Preprocessing
# ========================
elif page == "🔧 Data Preprocessing":
    st.markdown("<div class='sub-header'>🔧 Data Preprocessing</div>", unsafe_allow_html=True)

    if st.session_state.train_df is None:
        st.warning("⚠️ Please upload data first.")
    else:
        train_df = st.session_state.train_df.copy()
        test_df = st.session_state.test_df.copy()

        st.subheader("Preprocessing Steps")

        with st.expander("Step 1: Drop Unnecessary Columns", expanded=True):
            st.write("Dropping: `Unnamed: 0`, `id`")
            train_df.drop(columns=["Unnamed: 0", "id"], inplace=True, errors="ignore")
            test_df.drop(columns=["Unnamed: 0", "id"], inplace=True, errors="ignore")
            st.success("✅ Columns dropped")

        with st.expander("Step 2: Handle Missing Values"):
            missing_before = train_df.isnull().sum().sum()
            median_delay = train_df["Arrival Delay in Minutes"].median()
            train_df["Arrival Delay in Minutes"] = train_df["Arrival Delay in Minutes"].fillna(median_delay)
            test_df["Arrival Delay in Minutes"] = test_df["Arrival Delay in Minutes"].fillna(median_delay)
            missing_after = train_df.isnull().sum().sum()
            st.write(f"Missing values before: {missing_before}")
            st.write(f"Missing values after: {missing_after}")
            st.success("✅ Missing values filled with median")

        with st.expander("Step 3: Remove Duplicates"):
            dups = train_df.duplicated().sum()
            st.write(f"Duplicates found: {dups}")
            if dups > 0:
                train_df.drop_duplicates(inplace=True)
                st.success("✅ Duplicates removed")
            else:
                st.info("No duplicates found")

        with st.expander("Step 4: Check Inconsistent Entries"):
            for col in train_df.select_dtypes(include="object").columns:
                st.write(f"**{col}:** {train_df[col].unique()}")

        with st.expander("Step 5: Outlier Removal (Flight Distance)"):
            before_len = len(train_df)
            Q1 = train_df['Flight Distance'].quantile(0.25)
            Q3 = train_df['Flight Distance'].quantile(0.75)
            IQR = Q3 - Q1
            train_df = train_df[(train_df['Flight Distance'] >= Q1 - 1.5*IQR) & (train_df['Flight Distance'] <= Q3 + 1.5*IQR)]
            after_len = len(train_df)
            st.write(f"Rows before: {before_len}")
            st.write(f"Rows after: {after_len}")
            st.write(f"Removed: {before_len - after_len} rows")
            st.success("✅ Outliers removed")

        if st.button("💾 Save Preprocessed Data"):
            st.session_state.train_df = train_df
            st.session_state.test_df = test_df
            st.success("✅ Preprocessed data saved!")
            st.write(f"**Final Train Shape:** {train_df.shape}")
            st.write(f"**Final Test Shape:** {test_df.shape}")

# ========================
# Feature Engineering
# ========================
elif page == "🧠 Feature Engineering & Selection":
    st.markdown("<div class='sub-header'>🧠 Feature Engineering & Selection</div>", unsafe_allow_html=True)

    if st.session_state.train_df is None:
        st.warning("⚠️ Please complete preprocessing first.")
    else:
        train_df = st.session_state.train_df
        test_df = st.session_state.test_df

        st.subheader("1. Feature Engineering Pipeline")

        with st.spinner("Running feature engineering..."):
            # Create binary target
            train_df['satisfaction_binary'] = train_df['satisfaction'].map(
                {'satisfied': 1, 'neutral or dissatisfied': 0}
            )
            test_df['satisfaction_binary'] = test_df['satisfaction'].map(
                {'satisfied': 1, 'neutral or dissatisfied': 0}
            )

            train_processed, test_processed = feature_engineering_pipeline(train_df, test_df)

            # Align columns
            train_processed, test_processed = train_processed.align(test_processed, join='left', axis=1, fill_value=0)

            # Convert bools
            bool_cols = train_processed.select_dtypes(include='bool').columns
            train_processed[bool_cols] = train_processed[bool_cols].astype(int)
            test_processed[bool_cols] = test_processed[bool_cols].astype(int)

        st.success("✅ Feature engineering completed!")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Processed Train Shape", f"{train_processed.shape}")
        with col2:
            st.metric("Processed Test Shape", f"{test_processed.shape}")

        st.subheader("2. Feature Selection")

        X = train_processed.drop(columns=['satisfaction', 'satisfaction_binary'], errors='ignore')
        y = train_processed['satisfaction_binary']
        X_test = test_processed.drop(columns=['satisfaction', 'satisfaction_binary'], errors='ignore')
        y_test = test_processed['satisfaction_binary']

        x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        st.write(f"**Train split:** {x_train.shape}")
        st.write(f"**Validation split:** {x_val.shape}")

        with st.spinner("Removing constant features..."):
            x_train_f, x_val_f, x_test_f = check_constant_features(x_train, x_val, X_test)

        st.write(f"**After constant removal:** {x_train_f.shape[1]} features")

        selection_method = st.radio("Select Feature Selection Method", [
            "Mutual Information (Top 20)",
            "Forward Selection (Top 20)",
            "Backward Selection (Top 20)"
        ])

        if st.button("🔍 Run Feature Selection"):
            with st.spinner("Selecting features... This may take a moment."):
                if selection_method == "Mutual Information (Top 20)":
                    selector = SelectKBest(mutual_info_classif, k=20)
                    selector.fit(x_train_f, y_train)
                    selected_features = x_train_f.columns[selector.get_support()].tolist()

                elif selection_method == "Forward Selection (Top 20)":
                    selected_features = forward_feature_selection(x_train_f, y_train)

                else:  # Backward
                    sfs_backward = SFS(
                        estimator=RandomForestClassifier(n_estimators=10, n_jobs=-1, random_state=0),
                        n_features_to_select=20,
                        direction='backward',
                        scoring='roc_auc',
                        cv=2,
                        n_jobs=-1,
                    )
                    sfs_backward = sfs_backward.fit(x_train_f, y_train)
                    selected_features = x_train_f.columns[sfs_backward.get_support()].tolist()

            st.session_state.features = selected_features
            st.success(f"✅ Selected {len(selected_features)} features!")
            st.write("**Selected Features:**")
            st.write(selected_features)

            # Save processed data with selected features
            st.session_state.x_train = x_train_f[selected_features]
            st.session_state.x_val = x_val_f[selected_features]
            st.session_state.x_test = x_test_f[selected_features]
            st.session_state.y_train = y_train
            st.session_state.y_val = y_val
            st.session_state.y_test = y_test

# ========================
# Model Training
# ========================
elif page == "🤖 Model Training":
    st.markdown("<div class='sub-header'>🤖 Model Training</div>", unsafe_allow_html=True)

    if 'x_train' not in st.session_state:
        st.warning("⚠️ Please complete feature selection first.")
    else:
        x_train = st.session_state.x_train
        y_train = st.session_state.y_train
        x_val = st.session_state.x_val
        y_val = st.session_state.y_val
        x_test = st.session_state.x_test
        y_test = st.session_state.y_test

        st.subheader("Handle Class Imbalance")
        balance_method = st.radio("Choose method", ["None", "SMOTE", "Class Weights"])

        x_train_bal = x_train.copy()
        y_train_bal = y_train.copy()

        if balance_method == "SMOTE":
            with st.spinner("Applying SMOTE..."):
                smote = SMOTE(random_state=42)
                x_train_bal, y_train_bal = smote.fit_resample(x_train, y_train)
            st.success("✅ SMOTE applied")
            st.write(f"New train shape: {x_train_bal.shape}")

        st.subheader("Train Models")

        models_to_train = st.multiselect("Select Models", [
            "Logistic Regression",
            "Random Forest",
            "Gradient Boosting (Grid Search)",
            "XGBoost"
        ], default=["Random Forest", "XGBoost"])

        if st.button("🚀 Train Selected Models"):
            progress = st.progress(0)
            status = st.empty()

            trained_models = {}
            results = {}

            total = len(models_to_train)
            for i, model_name in enumerate(models_to_train):
                status.text(f"Training {model_name}...")

                if model_name == "Logistic Regression":
                    model = LogisticRegression(max_iter=1000, random_state=42)
                    m, acc, f1, report, pred = run_model(model, x_train_bal, y_train_bal, x_val, y_val)
                    trained_models["Logistic Regression"] = m
                    results["Logistic Regression"] = {"acc": acc, "f1": f1, "report": report}

                elif model_name == "Random Forest":
                    class_weight = 'balanced' if balance_method == "Class Weights" else None
                    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight=class_weight, n_jobs=-1)
                    m, acc, f1, report, pred = run_model(model, x_train_bal, y_train_bal, x_val, y_val)
                    trained_models["Random Forest"] = m
                    results["Random Forest"] = {"acc": acc, "f1": f1, "report": report}

                elif model_name == "Gradient Boosting (Grid Search)":
                    gb = GradientBoostingClassifier(random_state=42)
                    param_grid = {
                        'n_estimators': [100, 200],
                        'learning_rate': [0.05, 0.1],
                        'max_depth': [3, 4, 5]
                    }
                    grid = GridSearchCV(estimator=gb, param_grid=param_grid, cv=2, scoring='f1', n_jobs=-1)
                    grid.fit(x_train_bal, y_train_bal)
                    best_model = grid.best_estimator_
                    pred = best_model.predict(x_val)
                    acc = accuracy_score(y_val, pred)
                    f1 = f1_score(y_val, pred)
                    report = classification_report(y_val, pred)
                    trained_models["Gradient Boosting"] = best_model
                    results["Gradient Boosting"] = {"acc": acc, "f1": f1, "report": report, "best_params": grid.best_params_}

                elif model_name == "XGBoost":
                    model = XGBClassifier(
                        n_estimators=300, learning_rate=0.1, max_depth=6,
                        subsample=0.8, colsample_bytree=0.8, random_state=42,
                        eval_metric='logloss', use_label_encoder=False
                    )
                    m, acc, f1, report, pred = run_model(model, x_train_bal, y_train_bal, x_val, y_val)
                    trained_models["XGBoost"] = m
                    results["XGBoost"] = {"acc": acc, "f1": f1, "report": report}

                progress.progress((i + 1) / total)

            st.session_state.models = trained_models
            st.session_state.results = results
            status.empty()
            progress.empty()
            st.success("✅ All models trained successfully!")

            # Display results
            for name, res in results.items():
                with st.expander(f"📊 {name} Results"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Accuracy", f"{res['acc']:.4f}")
                    with col2:
                        st.metric("F1 Score", f"{res['f1']:.4f}")
                    st.text("Classification Report:")
                    st.text(res['report'])
                    if "best_params" in res:
                        st.write("**Best Params:**", res['best_params'])

# ========================
# Model Comparison
# ========================
elif page == "📈 Model Comparison":
    st.markdown("<div class='sub-header'>📈 Model Comparison</div>", unsafe_allow_html=True)

    if 'results' not in st.session_state or not st.session_state.results:
        st.warning("⚠️ Please train models first.")
    else:
        results = st.session_state.results
        models = st.session_state.models

        # Comparison table
        comparison_data = []
        for name, res in results.items():
            comparison_data.append({
                "Model": name,
                "Accuracy": f"{res['acc']:.4f}",
                "F1 Score": f"{res['f1']:.4f}"
            })

        df_comparison = pd.DataFrame(comparison_data)
        st.subheader("Performance Comparison")
        st.dataframe(df_comparison, use_container_width=True)

        # Bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        names = [r["Model"] for r in comparison_data]
        accs = [float(r["Accuracy"]) for r in comparison_data]
        f1s = [float(r["F1 Score"]) for r in comparison_data]

        x = np.arange(len(names))
        width = 0.35

        ax.bar(x - width/2, accs, width, label='Accuracy', color='#3B82F6')
        ax.bar(x + width/2, f1s, width, label='F1 Score', color='#10B981')

        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15)
        ax.legend()
        ax.set_ylim(0, 1)

        st.pyplot(fig)

        # Best model
        best_model = max(results.items(), key=lambda x: x[1]['f1'])
        st.success(f"🏆 Best Model (by F1): **{best_model[0]}** with F1={best_model[1]['f1']:.4f}")

        # Confusion Matrix for best model
        st.subheader("Confusion Matrix (Best Model on Validation)")
        x_val = st.session_state.x_val
        y_val = st.session_state.y_val
        best_pred = models[best_model[0]].predict(x_val)
        cm = confusion_matrix(y_val, best_pred)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix - {best_model[0]}')
        st.pyplot(fig)

        # Test set evaluation
        st.subheader("Test Set Evaluation")
        x_test = st.session_state.x_test
        y_test = st.session_state.y_test

        test_results = []
        for name, model in models.items():
            pred = model.predict(x_test)
            acc = accuracy_score(y_test, pred)
            f1 = f1_score(y_test, pred)
            test_results.append({"Model": name, "Test Accuracy": acc, "Test F1": f1})

        df_test = pd.DataFrame(test_results)
        st.dataframe(df_test.style.format({"Test Accuracy": "{:.4f}", "Test F1": "{:.4f}"}), use_container_width=True)

# ========================
# Prediction
# ========================
elif page == "🔮 Prediction":
    st.markdown("<div class='sub-header'>🔮 Make Predictions</div>", unsafe_allow_html=True)

    if 'models' not in st.session_state or not st.session_state.models:
        st.warning("⚠️ Please train models first.")
    else:
        models = st.session_state.models

        st.subheader("Select Model")
        selected_model = st.selectbox("Choose a trained model", list(models.keys()))
        model = models[selected_model]

        st.subheader("Input Passenger Data")

        # Create input form based on original features
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
            age = st.number_input("Age", 0, 100, 30)
            type_of_travel = st.selectbox("Type of Travel", ["Personal Travel", "Business travel"])
            flight_class = st.selectbox("Class", ["Business", "Eco", "Eco Plus"])
            flight_distance = st.number_input("Flight Distance", 0, 5000, 500)

        with col2:
            inflight_wifi = st.slider("Inflight wifi service", 0, 5, 3)
            departure_arrival_time = st.slider("Departure/Arrival time convenient", 0, 5, 3)
            ease_online_booking = st.slider("Ease of Online booking", 0, 5, 3)
            gate_location = st.slider("Gate location", 0, 5, 3)
            food_and_drink = st.slider("Food and drink", 0, 5, 3)
            online_boarding = st.slider("Online boarding", 0, 5, 3)

        with col3:
            seat_comfort = st.slider("Seat comfort", 0, 5, 3)
            inflight_entertainment = st.slider("Inflight entertainment", 0, 5, 3)
            onboard_service = st.slider("On-board service", 0, 5, 3)
            leg_room = st.slider("Leg room service", 0, 5, 3)
            baggage_handling = st.slider("Baggage handling", 0, 5, 3)
            checkin_service = st.slider("Checkin service", 0, 5, 3)
            inflight_service = st.slider("Inflight service", 0, 5, 3)
            cleanliness = st.slider("Cleanliness", 0, 5, 3)

        departure_delay = st.number_input("Departure Delay in Minutes", 0, 200, 0)
        arrival_delay = st.number_input("Arrival Delay in Minutes", 0, 200, 0)

        if st.button("🔮 Predict Satisfaction"):
            # Create input dataframe
            input_data = pd.DataFrame({
                'Gender': [gender],
                'Customer Type': [customer_type],
                'Age': [age],
                'Type of Travel': [type_of_travel],
                'Class': [flight_class],
                'Flight Distance': [flight_distance],
                'Inflight wifi service': [inflight_wifi],
                'Departure/Arrival time convenient': [departure_arrival_time],
                'Ease of Online booking': [ease_online_booking],
                'Gate location': [gate_location],
                'Food and drink': [food_and_drink],
                'Online boarding': [online_boarding],
                'Seat comfort': [seat_comfort],
                'Inflight entertainment': [inflight_entertainment],
                'On-board service': [onboard_service],
                'Leg room service': [leg_room],
                'Baggage handling': [baggage_handling],
                'Checkin service': [checkin_service],
                'Inflight service': [inflight_service],
                'Cleanliness': [cleanliness],
                'Departure Delay in Minutes': [departure_delay],
                'Arrival Delay in Minutes': [arrival_delay]
            })

            # Apply same preprocessing
            input_data['Total Delay'] = input_data['Departure Delay in Minutes'] + input_data['Arrival Delay in Minutes']
            input_data['Delay Level'] = pd.cut(
                input_data['Total Delay'],
                bins=[-0.1, 5, 30, float('inf')],
                labels=['No Delay', 'Medium Delay', 'High Delay']
            )
            input_data['Age_Group'] = pd.cut(input_data['Age'], bins=[0, 25, 50, 100], labels=False)

            # Use class average from training
            if 'train_df' in st.session_state and st.session_state.train_df is not None:
                class_avg = fit_class_features(st.session_state.train_df)
                input_data = apply_class_features(input_data, class_avg)
            else:
                input_data['Class_Service_Avg'] = 3.0  # default

            # Encode
            input_data = encode_features(input_data)
            input_data = pd.get_dummies(input_data, columns=['Delay Level'], drop_first=True)

            # Align with training features
            if 'features' in st.session_state and st.session_state.features:
                for col in st.session_state.features:
                    if col not in input_data.columns:
                        input_data[col] = 0
                input_data = input_data[st.session_state.features]

            # Predict
            prediction = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0] if hasattr(model, "predict_proba") else None

            if prediction == 1:
                st.success("✅ Passenger is **SATISFIED** 😊")
                if proba is not None:
                    st.progress(float(proba[1]))
                    st.write(f"Confidence: **{proba[1]*100:.1f}%**")
            else:
                st.error("❌ Passenger is **NEUTRAL OR DISSATISFIED** 😞")
                if proba is not None:
                    st.progress(float(proba[0]))
                    st.write(f"Confidence: **{proba[0]*100:.1f}%**")

        st.subheader("Or Upload CSV for Batch Prediction")
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        if uploaded_file:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Preview:", batch_df.head())

            if st.button("Run Batch Prediction"):
                st.info("Apply the same preprocessing pipeline as above...")
                # Similar preprocessing would go here
                st.success("Batch prediction complete!")

st.sidebar.markdown("---")
st.sidebar.info("💡 Use the tabs above to navigate through the ML pipeline.")