import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Flight Price Predictor", layout="wide")

st.title("✈️ Flight Price Prediction App")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # Drop unnecessary columns
    for col in ["Sno", "flight"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    TARGET_COL = "price"

    if TARGET_COL not in df.columns:
        st.error("Dataset must contain 'price' column")
    else:
        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]

        # Encoding
        X_encoded = pd.get_dummies(X, drop_first=True)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42
        )

        st.subheader("⚙️ Model Training")

        # Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)

        # Random Forest
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)

        # Metrics
        def evaluate(y_true, y_pred):
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            return mae, rmse, r2

        mae_lr, rmse_lr, r2_lr = evaluate(y_test, y_pred_lr)
        mae_rf, rmse_rf, r2_rf = evaluate(y_test, y_pred_rf)

        st.subheader("📈 Model Performance")

        col1, col2 = st.columns(2)

        with col1:
            st.write("### Linear Regression")
            st.write(f"MAE: {mae_lr:.2f}")
            st.write(f"RMSE: {rmse_lr:.2f}")
            st.write(f"R²: {r2_lr:.4f}")

        with col2:
            st.write("### Random Forest")
            st.write(f"MAE: {mae_rf:.2f}")
            st.write(f"RMSE: {rmse_rf:.2f}")
            st.write(f"R²: {r2_rf:.4f}")

        st.success("✅ Random Forest is usually better for this dataset")

        # Scatter Plot
        st.subheader("📉 Actual vs Predicted (Random Forest)")

        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred_rf, alpha=0.3)
        ax.plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()],
                "--")
        ax.set_xlabel("Actual Price")
        ax.set_ylabel("Predicted Price")
        st.pyplot(fig)

        # Feature Importance
        st.subheader("🔥 Feature Importance")

        feat_imp = pd.DataFrame({
            "feature": X_encoded.columns,
            "importance": rf.feature_importances_
        })

        feat_imp["orig"] = feat_imp["feature"].apply(lambda x: x.split("_")[0])
        agg_imp = feat_imp.groupby("orig")["importance"].sum().sort_values(ascending=False)

        st.bar_chart(agg_imp)

        # Prediction Section
        st.subheader("🎯 Make Prediction")

        user_input = {}

        for col in X.columns:
            if df[col].dtype == "object":
                user_input[col] = st.selectbox(col, df[col].unique())
            else:
                user_input[col] = st.number_input(col, float(df[col].min()), float(df[col].max()))

        if st.button("Predict Price"):
            input_df = pd.DataFrame([user_input])
            input_encoded = pd.get_dummies(input_df)

            # Align columns
            input_encoded = input_encoded.reindex(columns=X_encoded.columns, fill_value=0)

            prediction = rf.predict(input_encoded)[0]

            st.success(f"💰 Predicted Flight Price: ₹ {prediction:.2f}")
