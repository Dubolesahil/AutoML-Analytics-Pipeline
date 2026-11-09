
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    classification_report
)

st.set_page_config(
    page_title="AutoML Analytics Pipeline",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AutoML Analytics Pipeline")
st.write("Automated Data Cleaning, Machine Learning and Business Analytics")

# Upload dataset
uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV Dataset",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload your customer sales CSV dataset.")
    st.stop()

# Load dataset
df = pd.read_csv(uploaded_file)

required_columns = [
    'Customer_ID', 'Age', 'Gender', 'City',
    'Product_Category', 'Quantity', 'Unit_Price',
    'Discount_Percent', 'Marketing_Spend',
    'Previous_Purchases', 'Sales', 'Profit',
    'Purchase_Status'
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
    st.stop()

# Data cleaning
original_rows = len(df)
df = df.drop_duplicates().copy()

for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include=['object', 'string']).columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

# Feature engineering
df['Age_Group'] = pd.cut(
    df['Age'],
    bins=[0, 25, 35, 50, 65, 100],
    labels=['18-25', '26-35', '36-50', '51-65', '65+']
)

df['Customer_Type'] = np.where(
    df['Previous_Purchases'] > 0,
    'Returning Customer',
    'New Customer'
)

df['Gross_Order_Value'] = df['Quantity'] * df['Unit_Price']

df['Discount_Amount'] = (
    df['Gross_Order_Value'] * df['Discount_Percent'] / 100
)

df['Profit_Margin'] = np.where(
    df['Sales'] != 0,
    df['Profit'] / df['Sales'] * 100,
    0
)

# Dashboard KPIs
st.subheader("Business Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Sales", f"₹{df['Sales'].sum():,.2f}")
c2.metric("Total Profit", f"₹{df['Profit'].sum():,.2f}")
c3.metric("Customers", f"{df['Customer_ID'].nunique():,}")
c4.metric("Records", f"{len(df):,}")

st.success(
    f"Data cleaned successfully. "
    f"{original_rows - len(df)} duplicate rows removed."
)

# Dataset preview
with st.expander("View Dataset"):
    st.dataframe(df, use_container_width=True)

# Business visualizations
st.subheader("Business Analytics Dashboard")

col1, col2 = st.columns(2)

with col1:
    category_sales = df.groupby(
        'Product_Category'
    )['Sales'].sum().sort_values(ascending=False)

    st.write("Sales by Product Category")
    st.bar_chart(category_sales)

with col2:
    city_profit = df.groupby(
        'City'
    )['Profit'].sum().sort_values(ascending=False)

    st.write("Profit by City")
    st.bar_chart(city_profit)

st.write("Purchase Status Distribution")
st.bar_chart(df['Purchase_Status'].value_counts())

# AutoML
st.subheader("Automated Machine Learning")

target = st.selectbox(
    "Select Prediction Target",
    ["Sales", "Profit", "Purchase_Status"]
)

if st.button("Train and Compare Models"):

    X = df.drop(
        columns=[
            'Customer_ID', 'Sales', 'Profit',
            'Purchase_Status', 'Profit_Margin',
            'Discount_Amount'
        ],
        errors='ignore'
    )

    y = df[target]

    if y.nunique() < 2:
        st.error("The selected target must have at least two classes or values.")
        st.stop()

    if target == "Purchase_Status":

        class_counts = y.value_counts()

        if class_counts.min() < 2:
            st.error("Each class needs at least two records for stratified splitting.")
            st.stop()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        scoring_name = "Weighted F1 Score"

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        }

        results = []
        best_score = -1
        best_model = None
        best_predictions = None

        numeric_features = X_train.select_dtypes(
            include=np.number
        ).columns.tolist()

        categorical_features = X_train.select_dtypes(
            exclude=np.number
        ).columns.tolist()

        preprocessor = ColumnTransformer([
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),

            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ])

        for name, model in models.items():

            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])

            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)

            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(
                y_test,
                predictions,
                average='weighted',
                zero_division=0
            )

            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'Weighted F1 Score': f1
            })

            if f1 > best_score:
                best_score = f1
                best_model = pipeline
                best_predictions = predictions

        st.subheader("Classification Results")
        st.dataframe(pd.DataFrame(results), use_container_width=True)

        st.write("Best Model:", type(best_model.named_steps['model']).__name__)
        st.write("Classification Report")

        st.text(classification_report(
            y_test,
            best_predictions,
            zero_division=0
        ))

    else:

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42
        )

        numeric_features = X_train.select_dtypes(
            include=np.number
        ).columns.tolist()

        categorical_features = X_train.select_dtypes(
            exclude=np.number
        ).columns.tolist()

        preprocessor = ColumnTransformer([
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),

            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ])

        models = {
            "Linear Regression": LinearRegression(),

            "Random Forest": RandomForestRegressor(
                n_estimators=100,
                random_state=42
            ),

            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=100,
                random_state=42
            )
        }

        results = []
        best_rmse = float('inf')
        best_model = None

        for name, model in models.items():

            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])

            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)

            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)

            results.append({
                'Model': name,
                'MAE': mae,
                'RMSE': rmse,
                'R2 Score': r2
            })

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = pipeline
                best_predictions = predictions

        st.subheader("Regression Results")
        results_df = pd.DataFrame(results).sort_values('RMSE')
        st.dataframe(results_df, use_container_width=True)

        st.success(
            f"Best Model: {type(best_model.named_steps['model']).__name__}"
        )

        st.write("Actual vs Predicted Values")

        comparison = pd.DataFrame({
            'Actual': y_test.values,
            'Predicted': best_predictions
        })

        st.dataframe(comparison.head(20), use_container_width=True)

        st.scatter_chart(
            comparison,
            x='Actual',
            y='Predicted'
        )

    # Download business report
    report = f"""
AUTOML ANALYTICS PIPELINE REPORT

Total Records: {len(df)}
Total Customers: {df['Customer_ID'].nunique()}
Total Sales: {df['Sales'].sum():.2f}
Total Profit: {df['Profit'].sum():.2f}

Prediction Target: {target}

Best Model: {type(best_model.named_steps['model']).__name__}

Model Results:
{pd.DataFrame(results).to_string(index=False)}
"""

    st.download_button(
        label="Download Model Report",
        data=report,
        file_name="AutoML_Model_Report.txt",
        mime="text/plain"
    )

# Download cleaned dataset
st.download_button(
    label="Download Cleaned Dataset",
    data=df.to_csv(index=False).encode('utf-8'),
    file_name="cleaned_customer_sales.csv",
    mime="text/csv"
)

st.caption("AutoML Analytics Pipeline | Final Year Engineering Project")
