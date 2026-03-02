## Step 00 - Imports

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

st.set_page_config(
    page_title="Seoul Bike Sharing – Demand Dashboard",
    layout="centered",
    page_icon="🚲",
)

## Step 01 - Load data (robust path and encoding)
_data_dir = Path(__file__).parent
_csv_path = _data_dir / "SeoulBikeData.csv"
try:
    df = pd.read_csv(_csv_path, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(_csv_path, encoding="latin-1")

# Normalize column names (degree symbol may be corrupted)
_col_rename = {}
for c in df.columns:
    if "temperature" in c.lower() and "dew" not in c.lower() and "°" not in c and "(" in c:
        _col_rename[c] = "Temperature (°C)"
    elif "dew" in c.lower() and "°" not in c and "(" in c:
        _col_rename[c] = "Dew point temperature (°C)"
if _col_rename:
    df = df.rename(columns=_col_rename)

## Sidebar - three pages only (rubric)
st.sidebar.title("Seoul Bike Sharing 🚲")
page = st.sidebar.selectbox(
    "Select Page",
    ["Introduction 📘", "Visualization 📊", "Prediction 🎯"],
)

# Optional landing image (only if file exists)
_img_path = _data_dir / "bike.jpg"
if _img_path.exists():
    st.image(str(_img_path))
else:
    st.write("   ")
st.write("   ")
st.write("   ")

## Page 1 – Introduction (Business Case + Data Presentation)
if page == "Introduction 📘":
    st.subheader("01 Introduction 📘")

    st.markdown("##### 🎯 Business case")
    st.markdown(
        "**Problem:** A bike-sharing operator in Seoul needs to anticipate **hourly demand** "
        "to rebalance bikes across stations, plan maintenance, and optimize staffing."
    )
    st.markdown(
        "**Objective:** Use historical hourly data (weather and calendar) to explain and predict **Rented Bike Count**."
    )
    st.markdown("**Approach:** Explore data → visualize insights → predict demand with linear regression.")

    st.markdown("##### 📂 Dataset")
    st.markdown(
        "Source: [Kaggle – Seoul Bike Sharing Demand Prediction](https://www.kaggle.com/datasets/saurabhshahane/seoul-bike-sharing-demand-prediction/data?select=SeoulBikeData.csv). "
        "Hourly records with weather and calendar features."
    )
    n_rows, n_cols = df.shape
    st.caption(f"**Shape:** {n_rows:,} rows × {n_cols} columns")

    st.markdown("##### Data Preview")
    rows = st.slider("Select a number of rows to display", 5, 20, 5)
    st.dataframe(df.head(rows))

    st.markdown("##### Missing values")
    missing = df.isnull().sum()
    st.write(missing)
    if missing.sum() == 0:
        st.success("✅ No missing values found")
    else:
        st.warning("⚠️ You have missing values")

    st.markdown("##### 📈 Summary Statistics")
    if st.button("Show Describe Table"):
        st.dataframe(df.describe())
    with st.expander("Column descriptions"):
        st.markdown(
            "| Column | Description |\n"
            "|--------|-------------|\n"
            "| Date | Record date |\n"
            "| Rented Bike Count | **Target** – hourly rentals |\n"
            "| Hour | 0–23 |\n"
            "| Temperature / Dew point | °C |\n"
            "| Humidity(%) | Relative humidity |\n"
            "| Wind speed, Visibility | Weather |\n"
            "| Solar Radiation, Rainfall, Snowfall | Weather |\n"
            "| Seasons, Holiday, Functioning Day | Categorical |"
        )

## Page 2 – Data Visualization (Insights)
elif page == "Visualization 📊":
    st.subheader("02 Data Visualization 📊")

    tab_insights, tab_explore, tab_corr = st.tabs(
        ["Guided insights", "Explore", "Correlation Heatmap 🔥"]
    )

    with tab_insights:
        st.markdown("###### Demand by hour (average)")
        by_hour = df.groupby("Hour")["Rented Bike Count"].mean()
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.bar(by_hour.index, by_hour.values, color="steelblue", edgecolor="navy", alpha=0.8)
        ax1.set_xlabel("Hour of day")
        ax1.set_ylabel("Average Rented Bike Count")
        ax1.set_title("Hourly demand pattern (e.g. rush hours)")
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()
        st.caption("Peaks typically at morning and evening commute hours.")

        st.markdown("###### Demand by season (average)")
        by_season = df.groupby("Seasons")["Rented Bike Count"].mean()
        order = ["Spring", "Summer", "Autumn", "Winter"]
        by_season = by_season.reindex([s for s in order if s in by_season.index]).dropna()
        if by_season.size:
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.bar(by_season.index, by_season.values, color="coral", edgecolor="darkred", alpha=0.8)
            ax2.set_xlabel("Season")
            ax2.set_ylabel("Average Rented Bike Count")
            ax2.set_title("Demand by season")
            plt.xticks(rotation=15)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()
            st.caption("Higher demand in warmer seasons (Summer/Autumn).")

        st.markdown("###### Demand vs temperature")
        temp_col = [c for c in df.columns if "Temperature" in c and "Dew" not in c]
        temp_col = temp_col[0] if temp_col else df.select_dtypes(include=np.number).columns[0]
        scatter_df = df[[temp_col, "Rented Bike Count"]].dropna()
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        ax3.scatter(scatter_df[temp_col], scatter_df["Rented Bike Count"], alpha=0.3, s=10)
        ax3.set_xlabel(temp_col)
        ax3.set_ylabel("Rented Bike Count")
        ax3.set_title("Demand vs temperature")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()
        st.caption("Warmer, comfortable weather tends to increase rentals.")

        st.markdown("###### Demand: Holiday vs No Holiday")
        if "Holiday" in df.columns:
            by_holiday = df.groupby("Holiday")["Rented Bike Count"].mean()
            fig4, ax4 = plt.subplots(figsize=(6, 4))
            ax4.bar(by_holiday.index.astype(str), by_holiday.values, color=["steelblue", "coral"], edgecolor="black", alpha=0.8)
            ax4.set_xlabel("Holiday")
            ax4.set_ylabel("Average Rented Bike Count")
            ax4.set_title("Demand on Holiday vs No Holiday")
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close()
            st.caption("Compare average demand on holidays vs regular days.")

    with tab_explore:
        col_x = st.selectbox("Select X-axis variable", df.columns, index=0, key="x")
        col_y = st.selectbox("Select Y-axis variable", df.columns, index=1, key="y")
        numeric_cols = [c for c in [col_x, col_y] if c in df.select_dtypes(include=np.number).columns]
        if len(numeric_cols) == 2:
            st.bar_chart(df[[col_x, col_y]].sort_values(by=col_x), use_container_width=True)
            st.line_chart(df[[col_x, col_y]].sort_values(by=col_x), use_container_width=True)
        else:
            st.info("Choose two numeric columns for bar/line charts, or use Guided insights.")

    with tab_corr:
        df_numeric = df.select_dtypes(include=np.number)
        fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
        sns.heatmap(df_numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
        ax_corr.set_title("Correlation matrix (numeric columns)")
        plt.tight_layout()
        st.pyplot(fig_corr)
        plt.close()

## Page 3 – Prediction (Linear Regression)
elif page == "Prediction 🎯":
    st.subheader("04 Prediction with Linear Regression 🎯")

    df2 = df.dropna().copy()
    target_col = "Rented Bike Count"

    # Encode categoricals
    le = LabelEncoder()
    for col in ["Seasons", "Holiday", "Functioning Day"]:
        if col in df2.columns:
            df2[col] = le.fit_transform(df2[col].astype(str))

    feature_cols = [
        c for c in df2.columns
        if c != target_col and c != "Date"
    ]
    default_features = [c for c in feature_cols if c in df2.columns]

    features_selection = st.sidebar.multiselect(
        "Select Features (X)", feature_cols, default=default_features
    )
    selected_metrics = st.sidebar.multiselect(
        "Metrics to display",
        ["Mean Squared Error (MSE)", "Mean Absolute Error (MAE)", "R² Score"],
        default=["Mean Absolute Error (MAE)"],
    )

    if not features_selection:
        st.warning("Select at least one feature in the sidebar.")
    else:
        X = df2[features_selection]
        y = df2[target_col]
        st.dataframe(X.head())
        st.dataframe(y.head())

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = np.maximum(model.predict(X_test), 0)  # clip to 0: counts can't be negative

        if "Mean Squared Error (MSE)" in selected_metrics:
            mse = metrics.mean_squared_error(y_test, predictions)
            st.write(f"- **MSE** {mse:,.2f}")
        if "Mean Absolute Error (MAE)" in selected_metrics:
            mae = metrics.mean_absolute_error(y_test, predictions)
            st.write(f"- **MAE** {mae:,.2f}")
        if "R² Score" in selected_metrics:
            r2 = metrics.r2_score(y_test, predictions)
            st.write(f"- **R²** {r2:,.3f}")

        mae_val = metrics.mean_absolute_error(y_test, predictions)
        mse_val = metrics.mean_squared_error(y_test, predictions)
        r2_val = metrics.r2_score(y_test, predictions)
        st.success(f"Model performance (MAE): {np.round(mae_val, 2)} bikes per hour.")

        st.markdown("##### Driving variables (model coefficients)")
        coef_df = pd.DataFrame({
            "Feature": features_selection,
            "Coefficient": model.coef_,
        })
        coef_df = coef_df.reindex(coef_df["Coefficient"].abs().sort_values(ascending=False).index)
        st.dataframe(coef_df.round(2), use_container_width=True)
        st.caption("Larger absolute coefficient = stronger effect on predicted demand. Positive: more of that feature → higher demand.")

        fig, ax = plt.subplots()
        ax.scatter(y_test, predictions, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--r", linewidth=2)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted (Rented Bike Count)")
        st.pyplot(fig)
        plt.close()
        st.caption("Predictions are clipped at 0 because linear regression can output negative values; bike counts cannot be negative.")

        st.markdown(
            "**How this solves the problem:** Expected hourly demand from the model can be used "
            "for rebalancing bikes across stations, planning maintenance windows, and staffing decisions."
        )

        st.markdown("---")
        st.markdown("##### Conclusion")
        st.markdown(
            f"On the test set (20% of the data), the model has **MAE = {mae_val:,.0f}** bikes per hour, "
            f"**MSE = {mse_val:,.0f}**, and **R² = {r2_val:.3f}**. "
            f"On average, predictions are off by about {mae_val:,.0f} rentals per hour; "
            f"R² indicates that the chosen features explain roughly {max(0, r2_val) * 100:.1f}% of the variance in demand. "
            "The coefficients above show which variables drive predictions most. "
            "Operators can use these forecasts for rebalancing, maintenance, and staffing, with the reported MAE as a guide to expected error."
        )
