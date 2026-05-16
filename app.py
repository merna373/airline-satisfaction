import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, SelectKBest, SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Airline Satisfaction Analyzer", layout="wide")
st.title("✈️ Airline Passenger Satisfaction Dashboard")
st.markdown("---")

# ------------------ HELPER FUNCTIONS ------------------
def add_binary_target(df):
    """Add satisfaction_binary column from satisfaction."""
    if 'satisfaction' in df.columns and 'satisfaction_binary' not in df.columns:
        df['satisfaction_binary'] = df['satisfaction'].map(
            {'satisfied': 1, 'neutral or dissatisfied': 0}
        )
    return df

# ------------------ CACHE DATA LOADING ------------------
@st.cache_data
def load_data(uploaded_train, uploaded_test):
    if uploaded_train is not None:
        train = pd.read_csv(uploaded_train)
    else:
        train = pd.read_csv("train.csv")  
    if uploaded_test is not None:
        test = pd.read_csv(uploaded_test)
    else:
        test = pd.read_csv("test.csv")
    
    for df in [train, test]:
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)
        if 'id' in df.columns:
            df.drop(columns=['id'], inplace=True)
    
    train = add_binary_target(train)
    test = add_binary_target(test)
    
    return train, test

# ------------------ FEATURE ENGINEERING (same as notebook) ------------------
def create_new_features(df):
    df = df.copy()
    if 'Delay Group' in df.columns:
        df.drop(columns=['Delay Group'], inplace=True, errors='ignore')
    df['Total Delay'] = df['Departure Delay in Minutes'].fillna(0) + df['Arrival Delay in Minutes'].fillna(0)
    df['Delay Level'] = pd.cut(df['Total Delay'], bins=[-0.1, 5, 30, float('inf')],
                               labels=['No Delay', 'Medium Delay', 'High Delay'])
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 25, 50, 100], labels=False)
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

    class_avg = train_df.groupby('Class')['Inflight service'].mean()
    train_df['Class_Service_Avg'] = train_df['Class'].map(class_avg)
    test_df['Class_Service_Avg'] = test_df['Class'].map(class_avg)
    train_df = encode_features(train_df)
    test_df = encode_features(test_df)

    train_df = pd.get_dummies(train_df, columns=['Delay Level'], drop_first=True)
    test_df = pd.get_dummies(test_df, columns=['Delay Level'], drop_first=True)

    train_df, test_df = train_df.align(test_df, join='left', axis=1, fill_value=0)
    return train_df, test_df, class_avg

# ------------------ MAIN APP ------------------
def main():
    st.sidebar.header("📂 Data Upload")
    uploaded_train = st.sidebar.file_uploader("Upload train.csv", type="csv")
    uploaded_test = st.sidebar.file_uploader("Upload test.csv", type="csv")

    if uploaded_train is None or uploaded_test is None:
        st.info("Please upload both train.csv and test.csv files to begin.")
        st.stop()

    train_df, test_df = load_data(uploaded_train, uploaded_test)

    st.sidebar.success("✅ Data loaded successfully")
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Pipeline Options")
    do_preprocessing = st.sidebar.checkbox("Apply Feature Engineering", value=True)
    handle_missing = st.sidebar.checkbox("Fill missing Arrival Delay (median)", value=True)
    remove_outliers = st.sidebar.checkbox("Remove outliers (Flight Distance IQR)", value=False)

    # ------------------ TABS (now 5 tabs) ------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Overview", "📈 EDA", "🔧 Preprocessing", "🤖 Modeling", "🔮 Predict"])

    with tab1:
        st.subheader("Train Data")
        st.write(f"Shape: {train_df.shape}")
        st.dataframe(train_df.head())
        st.subheader("Test Data")
        st.write(f"Shape: {test_df.shape}")
        st.dataframe(test_df.head())
        st.subheader("Missing Values (Train)")
        missing = train_df.isnull().sum()
        st.write(missing[missing > 0] if any(missing > 0) else "No missing values")

    # ------------------ EDA (static plots) ------------------
    with tab2:
        st.subheader("Exploratory Data Analysis")
        plot_type = st.selectbox("Choose plot", ["Stacked Bar (Gender)", "Stacked Bar (Class)", "Stacked Bar (Customer Type)",
                                                 "Stacked Bar (Travel Type)", "Scatter (Age vs Distance)", "Scatter (Delays)",
                                                 "Correlation Heatmap", "Satisfaction Pie", "Class Pie", "Age Histogram",
                                                 "Flight Distance Histogram", "Boxplot Age", "Boxplot Departure Delay"])
        fig, ax = plt.subplots(figsize=(8,5))
        if plot_type == "Stacked Bar (Gender)":
            pivot = train_df.groupby(["Gender", "satisfaction"]).size().unstack()
            pivot.plot(kind="bar", stacked=True, ax=ax)
            ax.set_title("Satisfaction by Gender")
        elif plot_type == "Stacked Bar (Class)":
            pivot = train_df.groupby(["Class", "satisfaction"]).size().unstack()
            pivot.plot(kind="bar", stacked=True, ax=ax)
        elif plot_type == "Stacked Bar (Customer Type)":
            pivot = train_df.groupby(["Customer Type", "satisfaction"]).size().unstack()
            pivot.plot(kind="bar", stacked=True, ax=ax)
        elif plot_type == "Stacked Bar (Travel Type)":
            pivot = train_df.groupby(["Type of Travel", "satisfaction"]).size().unstack()
            pivot.plot(kind="bar", stacked=True, ax=ax)
        elif plot_type == "Scatter (Age vs Distance)":
            ax.scatter(train_df["Age"], train_df["Flight Distance"], alpha=0.4)
            ax.set_xlabel("Age"); ax.set_ylabel("Flight Distance")
        elif plot_type == "Scatter (Delays)":
            ax.scatter(train_df["Departure Delay in Minutes"], train_df["Arrival Delay in Minutes"], alpha=0.4)
            ax.set_xlabel("Departure Delay"); ax.set_ylabel("Arrival Delay")
        elif plot_type == "Correlation Heatmap":
            temp = train_df.copy()
            temp["satisfaction"] = temp["satisfaction"].map({"satisfied":1, "neutral or dissatisfied":0})
            corr = temp.corr(numeric_only=True)
            sns.heatmap(corr, cmap="coolwarm", ax=ax)
        elif plot_type == "Satisfaction Pie":
            train_df["satisfaction"].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
            ax.set_ylabel("")
        elif plot_type == "Class Pie":
            train_df["Class"].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
            ax.set_ylabel("")
        elif plot_type == "Age Histogram":
            sns.histplot(train_df["Age"], bins=20, kde=True, ax=ax)
        elif plot_type == "Flight Distance Histogram":
            sns.histplot(train_df["Flight Distance"], bins=30, kde=True, ax=ax)
        elif plot_type == "Boxplot Age":
            sns.boxplot(x=train_df["Age"], ax=ax)
        elif plot_type == "Boxplot Departure Delay":
            sns.boxplot(x=train_df["Departure Delay in Minutes"], ax=ax)
        st.pyplot(fig)

    # ------------------ PREPROCESSING TAB (only data preparation) ------------------
    with tab3:
        st.subheader("Data Preprocessing")
        if handle_missing:
            median_delay = train_df["Arrival Delay in Minutes"].median()
            train_df["Arrival Delay in Minutes"].fillna(median_delay, inplace=True)
            test_df["Arrival Delay in Minutes"].fillna(median_delay, inplace=True)
            st.write("✅ Missing values in 'Arrival Delay in Minutes' filled with median.")
        if remove_outliers:
            Q1 = train_df['Flight Distance'].quantile(0.25)
            Q3 = train_df['Flight Distance'].quantile(0.75)
            IQR = Q3 - Q1
            train_df = train_df[(train_df['Flight Distance'] >= Q1 - 1.5*IQR) & (train_df['Flight Distance'] <= Q3 + 1.5*IQR)]
            st.write(f"✅ Outliers removed. New train shape: {train_df.shape}")

        if do_preprocessing:
            with st.spinner("Applying feature engineering..."):
                train_proc, test_proc, class_avg = feature_engineering_pipeline(train_df, test_df)
            st.success("Feature engineering completed!")
            st.session_state['class_avg'] = class_avg
            st.write("Processed train shape:", train_proc.shape)
            st.write("Processed test shape:", test_proc.shape)
            st.dataframe(train_proc.head())
        else:
            train_proc, test_proc = train_df, test_df
            class_avg = train_df.groupby('Class')['Inflight service'].mean()
            st.session_state['class_avg'] = class_avg
            st.warning("Feature engineering skipped. Please be aware that models may not perform well.")

        st.session_state['train_proc'] = train_proc
        st.session_state['test_proc'] = test_proc
        st.success("Preprocessed data saved. Go to the Modeling tab to train models.")

    # ------------------ MODELING TAB (Model Configuration + Training + Evaluation) ------------------
    with tab4:
        st.subheader("Model Configuration & Training")
        
        if 'train_proc' not in st.session_state:
            st.warning("Please run preprocessing first (in the Preprocessing tab).")
            st.stop()
        
        train_proc = st.session_state['train_proc'].copy()
        test_proc = st.session_state['test_proc'].copy()
        
        if 'satisfaction_binary' not in train_proc.columns:
            train_proc = add_binary_target(train_proc)
        if 'satisfaction_binary' not in test_proc.columns:
            test_proc = add_binary_target(test_proc)
        
        X = train_proc.drop(columns=['satisfaction', 'satisfaction_binary'])
        st.session_state['all_columns'] = X.columns
        y = train_proc['satisfaction_binary']
        X_test = test_proc.drop(columns=['satisfaction', 'satisfaction_binary'])
        y_test = test_proc['satisfaction_binary']
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        st.subheader("Feature Selection")
        fs_method = st.selectbox("Select method", ["None", "Variance Threshold", "Mutual Information (k=20)",
                                                   "Forward Selection (k=20)", "Backward Selection (k=20)"])
        if fs_method == "Variance Threshold":
            selector = VarianceThreshold(threshold=0)
            X_train_fs = selector.fit_transform(X_train)
            X_val_fs = selector.transform(X_val)
            X_test_fs = selector.transform(X_test)
            cols = X_train.columns[selector.get_support()]
            st.write(f"Selected {len(cols)} non-constant features")
        elif fs_method == "Mutual Information (k=20)":
            selector = SelectKBest(mutual_info_classif, k=20)
            X_train_fs = selector.fit_transform(X_train, y_train)
            X_val_fs = selector.transform(X_val)
            X_test_fs = selector.transform(X_test)
            cols = X_train.columns[selector.get_support()]
            st.write("Selected features:", list(cols))
        elif fs_method == "Forward Selection (k=20)":
            with st.spinner("Forward selection may take a while..."):
                sfs = SequentialFeatureSelector(RandomForestClassifier(n_estimators=10, random_state=0),
                                                n_features_to_select=20, direction='forward', scoring='roc_auc', cv=2, n_jobs=-1)
                sfs.fit(X_train, y_train)
                cols = X_train.columns[sfs.get_support()]
                X_train_fs = sfs.transform(X_train)
                X_val_fs = sfs.transform(X_val)
                X_test_fs = sfs.transform(X_test)
            st.write("Forward selected features:", list(cols))
        elif fs_method == "Backward Selection (k=20)":
            with st.spinner("Backward selection may take a while..."):
                sfs = SequentialFeatureSelector(RandomForestClassifier(n_estimators=10, random_state=0),
                                                n_features_to_select=20, direction='backward', scoring='roc_auc', cv=2, n_jobs=-1)
                sfs.fit(X_train, y_train)
                cols = X_train.columns[sfs.get_support()]
                X_train_fs = sfs.transform(X_train)
                X_val_fs = sfs.transform(X_val)
                X_test_fs = sfs.transform(X_test)
            st.write("Backward selected features:", list(cols))
        else:
            X_train_fs, X_val_fs, X_test_fs = X_train, X_val, X_test
            cols = X_train.columns
        
        st.subheader("Handle Imbalance")
        imbalance_method = st.selectbox("Method", ["None", "SMOTE", "Class Weight (for RF/XGB/LR)"])
        class_weight = None
        if imbalance_method == "SMOTE":
            sm = SMOTE(random_state=42)
            X_train_fs, y_train = sm.fit_resample(X_train_fs, y_train)
            st.write("SMOTE applied. New training set size:", X_train_fs.shape[0])
        elif imbalance_method == "Class Weight (for RF/XGB/LR)":
            class_weight = 'balanced'
        
        st.subheader("Model Configuration")
        model_choice = st.selectbox("Choose model", ["Logistic Regression", "Random Forest", "XGBoost", "Gradient Boosting"])
        params = {}
        if model_choice == "Logistic Regression":
            C = st.number_input("C (inverse regularization)", 0.01, 10.0, 1.0)
            params['C'] = C
        elif model_choice == "Random Forest":
            n_estimators = st.slider("n_estimators", 50, 300, 100)
            max_depth = st.slider("max_depth", 3, 20, 10)
            params['n_estimators'] = n_estimators
            params['max_depth'] = max_depth
        elif model_choice == "XGBoost":
            n_estimators = st.slider("n_estimators", 50, 300, 100)
            learning_rate = st.number_input("learning_rate", 0.01, 0.5, 0.1)
            max_depth = st.slider("max_depth", 3, 10, 6)
            params['n_estimators'] = n_estimators
            params['learning_rate'] = learning_rate
            params['max_depth'] = max_depth
        elif model_choice == "Gradient Boosting":
            n_estimators = st.slider("n_estimators", 50, 300, 100)
            learning_rate = st.number_input("learning_rate", 0.01, 0.5, 0.1)
            max_depth = st.slider("max_depth", 3, 10, 5)
            params['n_estimators'] = n_estimators
            params['learning_rate'] = learning_rate
            params['max_depth'] = max_depth
        
        train_button = st.button("🚀 Train Model")
        if train_button:

            if model_choice == "Logistic Regression":
                model = LogisticRegression(max_iter=1000, random_state=42, C=params.get('C',1.0), class_weight=class_weight)
            elif model_choice == "Random Forest":
                model = RandomForestClassifier(random_state=42, class_weight=class_weight, **params)
            elif model_choice == "XGBoost":
                model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', **params)
                if class_weight == 'balanced':
                    neg = (y_train == 0).sum()
                    pos = (y_train == 1).sum()
                    if pos > 0:
                        model.set_params(scale_pos_weight=neg / pos)
            elif model_choice == "Gradient Boosting":
                model = GradientBoostingClassifier(random_state=42, **params)
            
            with st.spinner("Training model..."):
                model.fit(X_train_fs, y_train)
                y_pred_val = model.predict(X_val_fs)
                y_pred_test = model.predict(X_test_fs)
            
            acc_val = accuracy_score(y_val, y_pred_val)
            f1_val = f1_score(y_val, y_pred_val)
            acc_test = accuracy_score(y_test, y_pred_test)
            f1_test = f1_score(y_test, y_pred_test)
            
            st.success("Training completed!")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Validation Accuracy", f"{acc_val:.3f}")
                st.metric("Validation F1", f"{f1_val:.3f}")
            with col2:
                st.metric("Test Accuracy", f"{acc_test:.3f}")
                st.metric("Test F1", f"{f1_test:.3f}")
            
            st.subheader("Classification Report (Test Set)")
            report = classification_report(y_test, y_pred_test, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())
            

            cm = confusion_matrix(y_test, y_pred_test)
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
            ax_cm.set_xlabel('Predicted'); ax_cm.set_ylabel('Actual')
            st.pyplot(fig_cm)
            

            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test_fs)[:,1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                fig_roc, ax_roc = plt.subplots()
                ax_roc.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
                ax_roc.plot([0,1],[0,1],'k--')
                ax_roc.set_xlabel('False Positive Rate'); ax_roc.set_ylabel('True Positive Rate')
                ax_roc.legend()
                st.pyplot(fig_roc)
            

            st.session_state['trained_model'] = model
            st.session_state['model_cols'] = cols
            st.session_state['fs_method'] = fs_method
            if fs_method in ["Variance Threshold", "Mutual Information (k=20)"]:
                st.session_state['selector'] = selector
            else:
                st.session_state['selector'] = None
            if fs_method in ["Forward Selection (k=20)", "Backward Selection (k=20)"]:
                st.session_state['sfs'] = sfs
            else:
                st.session_state['sfs'] = None

    # ------------------ PREDICTION TAB (unchanged) ------------------
    with tab5:
        st.subheader("Make a New Prediction")
        if 'trained_model' not in st.session_state:
            st.warning("Please train a model first (in the Modeling tab).")
        else:
            st.markdown("Enter passenger details to predict satisfaction (satisfied / neutral or dissatisfied)")
            
            col1, col2 = st.columns(2)
            with col1:
                gender = st.selectbox("Gender", ["Male", "Female"])
                customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
                age = st.slider("Age", 0, 100, 30)
                travel_type = st.selectbox("Type of Travel", ["Personal Travel", "Business travel"])
                flight_class = st.selectbox("Class", ["Business", "Eco", "Eco Plus"])
                flight_distance = st.number_input("Flight Distance", min_value=0, value=500)
            with col2:
                inflight_wifi = st.slider("Inflight wifi service", 0, 5, 3)
                departure_arrival_convenient = st.slider("Departure/Arrival time convenient", 0, 5, 3)
                online_boarding = st.slider("Online boarding", 0, 5, 3)
                seat_comfort = st.slider("Seat comfort", 0, 5, 3)
                inflight_entertainment = st.slider("Inflight entertainment", 0, 5, 3)
                on_board_service = st.slider("On-board service", 0, 5, 3)
                leg_room = st.slider("Leg room service", 0, 5, 3)
                baggage_handling = st.slider("Baggage handling", 0, 5, 3)
                checkin_service = st.slider("Checkin service", 0, 5, 3)
                inflight_service = st.slider("Inflight service", 0, 5, 3)
                cleanliness = st.slider("Cleanliness", 0, 5, 3)
                departure_delay = st.number_input("Departure Delay (minutes)", value=0)
                arrival_delay = st.number_input("Arrival Delay (minutes)", value=0)
            
            if st.button("Predict Satisfaction"):
                input_data = {
                    "Gender": gender,
                    "Customer Type": customer_type,
                    "Age": age,
                    "Type of Travel": travel_type,
                    "Class": flight_class,
                    "Flight Distance": flight_distance,
                    "Inflight wifi service": inflight_wifi,
                    "Departure/Arrival time convenient": departure_arrival_convenient,
                    "Online boarding": online_boarding,
                    "Seat comfort": seat_comfort,
                    "Inflight entertainment": inflight_entertainment,
                    "On-board service": on_board_service,
                    "Leg room service": leg_room,
                    "Baggage handling": baggage_handling,
                    "Checkin service": checkin_service,
                    "Inflight service": inflight_service,
                    "Cleanliness": cleanliness,
                    "Departure Delay in Minutes": departure_delay,
                    "Arrival Delay in Minutes": arrival_delay,
                }
                input_df = pd.DataFrame([input_data])
                
                input_df = create_new_features(input_df)
                if 'class_avg' not in st.session_state:
                    st.error("Class average not found. Please re-run preprocessing.")
                    st.stop()
                class_avg = st.session_state['class_avg']
                input_df['Class_Service_Avg'] = input_df['Class'].map(class_avg)
                input_df = encode_features(input_df)
                input_df = pd.get_dummies(input_df, columns=['Delay Level'], drop_first=True)
                
                all_columns = st.session_state['all_columns']
                for col in all_columns:
                    if col not in input_df.columns:
                        input_df[col] = 0
                input_df = input_df[all_columns]

                fs_method = st.session_state.get('fs_method', 'None')
                if fs_method == "Variance Threshold" and st.session_state['selector'] is not None:
                    input_transformed = st.session_state['selector'].transform(input_df)
                elif fs_method in ["Mutual Information (k=20)", "Forward Selection (k=20)", "Backward Selection (k=20)"]:
                    if fs_method == "Mutual Information (k=20)" and st.session_state['selector'] is not None:
                        input_transformed = st.session_state['selector'].transform(input_df)
                    elif st.session_state['sfs'] is not None:
                        input_transformed = st.session_state['sfs'].transform(input_df)
                    else:
                        input_transformed = input_df.values
                else:
                    input_transformed = input_df.values
                
                model = st.session_state['trained_model']
                pred = model.predict(input_transformed)[0]
                proba = model.predict_proba(input_transformed)[0] if hasattr(model, "predict_proba") else [0,0]
                result = "Satisfied" if pred == 1 else "Neutral or Dissatisfied"
                st.success(f"Prediction: **{result}**")
                st.write(f"Confidence: Satisfied = {proba[1]:.2f}, Not Satisfied = {proba[0]:.2f}")

if __name__ == "__main__":
    main()