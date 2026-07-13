import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Sales Forecasting & Demand Intelligence", layout="wide")


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_train():
    df = pd.read_csv("train.csv", encoding="latin1")
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Order Date", "Sales"]).sort_values("Order Date")
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Quarter"] = df["Order Date"].dt.quarter
    return df


@st.cache_data
def load_forecast_table():
    fc = pd.read_csv("forecast_table.csv")
    fc["Month"] = pd.to_datetime(fc["Month"])
    return fc


@st.cache_data
def load_anomaly_report():
    an = pd.read_csv("anomaly_report.csv")
    an["Order Date"] = pd.to_datetime(an["Order Date"])
    return an


@st.cache_data
def load_cluster_table():
    return pd.read_csv("cluster_table.csv")


df = load_train()
forecast_table = load_forecast_table()
anomaly_report = load_anomaly_report()
cluster_table = load_cluster_table()


def get_monthly_sales(data, category=None, region=None):
    temp = data.copy()
    if category and category != "All":
        temp = temp[temp["Category"] == category]
    if region and region != "All":
        temp = temp[temp["Region"] == region]
    monthly = temp.set_index("Order Date").resample("MS")["Sales"].sum().reset_index()
    monthly.columns = ["ds", "y"]
    return monthly


def get_weekly_sales(data):
    weekly = data.set_index("Order Date").resample("W")["Sales"].sum().reset_index()
    weekly.columns = ["ds", "y"]
    return weekly


def backtest_mae_rmse(monthly, test_months=3):
    """Quick XGBoost lag-feature backtest: train on all but the last `test_months`,
    predict them, compare to actuals. Used only to report MAE/RMSE for the
    Forecast Explorer page."""
    from xgboost import XGBRegressor

    if len(monthly) < test_months + 9:
        return None, None

    data = monthly.copy()
    data["month"] = data["ds"].dt.month
    data["quarter"] = data["ds"].dt.quarter
    data["lag1"] = data["y"].shift(1)
    data["lag2"] = data["y"].shift(2)
    data["lag3"] = data["y"].shift(3)
    data["roll3"] = data["y"].rolling(3).mean()

    train_full = data.iloc[:-test_months].dropna()
    test_actual = data.iloc[-test_months:]["y"].values

    features = ["month", "quarter", "lag1", "lag2", "lag3", "roll3"]
    model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05)
    model.fit(train_full[features], train_full["y"])

    history = monthly.iloc[:-test_months].copy()
    preds = []
    for _ in range(test_months):
        last3 = history.tail(3)["y"].values
        next_date = history["ds"].iloc[-1] + pd.DateOffset(months=1)
        row = pd.DataFrame([{
            "month": next_date.month,
            "quarter": (next_date.month - 1) // 3 + 1,
            "lag1": last3[-1], "lag2": last3[-2], "lag3": last3[-3],
            "roll3": np.mean(last3),
        }])
        pred = model.predict(row[features])[0]
        history = pd.concat([history, pd.DataFrame({"ds": [next_date], "y": [pred]})], ignore_index=True)
        preds.append(pred)

    preds = np.array(preds)
    mae = np.mean(np.abs(test_actual - preds))
    rmse = np.sqrt(np.mean((test_actual - preds) ** 2))
    return mae, rmse


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.title("Sales Forecasting & Demand Intelligence")
page = st.sidebar.radio(
    "Navigate",
    ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Demand Segments"],
)

# ============================================================
# PAGE 1 — SALES OVERVIEW
# ============================================================
if page == "Sales Overview":
    st.title("Sales Overview Dashboard")

    col1, col2 = st.columns(2)
    with col1:
        region_filter = st.selectbox("Filter by Region", ["All"] + sorted(df["Region"].unique()))
    with col2:
        category_filter = st.selectbox("Filter by Category", ["All"] + sorted(df["Category"].unique()))

    filtered = df.copy()
    if region_filter != "All":
        filtered = filtered[filtered["Region"] == region_filter]
    if category_filter != "All":
        filtered = filtered[filtered["Category"] == category_filter]

    yearly = filtered.groupby("Year")["Sales"].sum()
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.bar(yearly.index.astype(str), yearly.values, color="#4C72B0")
    ax1.set_title("Total Sales by Year")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Sales")
    st.pyplot(fig1)
    plt.close(fig1)

    monthly = get_monthly_sales(filtered)
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(monthly["ds"], monthly["y"], marker="o", color="#DD8452")
    ax2.set_title("Monthly Sales Trend")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Sales")
    st.pyplot(fig2)
    plt.close(fig2)

    col3, col4 = st.columns(2)
    with col3:
        region_sales = filtered.groupby("Region")["Sales"].sum()
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        ax3.bar(region_sales.index, region_sales.values, color="#55A868")
        ax3.set_title("Sales by Region")
        ax3.set_ylabel("Sales")
        st.pyplot(fig3)
        plt.close(fig3)
    with col4:
        cat_sales = filtered.groupby("Category")["Sales"].sum()
        fig4, ax4 = plt.subplots(figsize=(5, 4))
        ax4.pie(cat_sales.values, labels=cat_sales.index, autopct="%1.1f%%",
                colors=["#4C72B0", "#DD8452", "#55A868"])
        ax4.set_title("Sales by Category")
        st.pyplot(fig4)
        plt.close(fig4)

# ============================================================
# PAGE 2 — FORECAST EXPLORER
# ============================================================
elif page == "Forecast Explorer":
    st.title("Forecast Explorer")
    st.caption("Forecast values come from the best model identified in Task 3/4 (forecast_table.csv).")

    segment_cols = [c for c in forecast_table.columns if c != "Month"]

    col1, col2 = st.columns(2)
    with col1:
        segment = st.selectbox("Select Category or Region", segment_cols)
    with col2:
        horizon = st.select_slider("Forecast Horizon (months ahead)", options=[1, 2, 3], value=3)

    # map the forecast column name back to the raw Category / Region field
    if segment in ["Furniture", "Technology", "Office Supplies"]:
        monthly_hist = get_monthly_sales(df, category=segment)
    else:
        region_name = segment.replace(" Region", "")
        monthly_hist = get_monthly_sales(df, region=region_name)

    future = forecast_table[["Month", segment]].head(horizon).rename(columns={segment: "y"})

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(monthly_hist["ds"], monthly_hist["y"], marker="o", label="Actual", color="#4C72B0")
    ax.plot(future["Month"], future["y"], marker="o", linestyle="--", label="Forecast", color="#C44E52")
    ax.axvline(monthly_hist["ds"].max(), color="gray", linestyle=":", linewidth=1)
    ax.set_title(f"Forecast — {segment}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Sales")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader(f"Forecasted values — next {horizon} month(s)")
    st.dataframe(future.rename(columns={"Month": "Month", "y": "Forecasted Sales"}))

    st.subheader("Model Performance")
    with st.spinner("Backtesting on the last 3 known months..."):
        mae, rmse = backtest_mae_rmse(monthly_hist, test_months=3)
    if mae is not None:
        c1, c2 = st.columns(2)
        c1.metric("MAE", f"{mae:,.2f}")
        c2.metric("RMSE", f"{rmse:,.2f}")
        st.caption("MAE/RMSE computed by holding out the last 3 known months and backtesting.")
    else:
        st.warning("Not enough history for this segment to backtest reliably.")

# ============================================================
# PAGE 3 — ANOMALY REPORT
# ============================================================
elif page == "Anomaly Report":
    st.title("Anomaly Report")

    weekly = get_weekly_sales(df)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(weekly["ds"], weekly["y"], color="#4C72B0", linewidth=1, label="Weekly Sales")
    ax.scatter(anomaly_report["Order Date"], anomaly_report["Sales"],
               color="red", marker="x", s=80, label="Detected Anomaly", zorder=5)
    ax.set_title("Weekly Sales — Detected Anomalies")
    ax.set_xlabel("Week")
    ax.set_ylabel("Sales")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Detected Anomalies")
    st.dataframe(
        anomaly_report.rename(columns={"Order Date": "Week", "Sales": "Sales", "Possible Explanation": "Likely Cause"})
    )

    st.info(f"{len(anomaly_report)} anomalous week(s) detected across the 4-year history.")

# ============================================================
# PAGE 4 — PRODUCT DEMAND SEGMENTS
# ============================================================
else:
    st.title("Product Demand Segments")

    fig, ax = plt.subplots(figsize=(8, 6))
    clusters = sorted(cluster_table["Cluster"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))
    for c, color in zip(clusters, colors):
        sub = cluster_table[cluster_table["Cluster"] == c]
        ax.scatter(sub["PC1"], sub["PC2"], color=color, s=90, label=f"Cluster {c}")
        for _, row in sub.iterrows():
            ax.annotate(row["Sub-Category"], (row["PC1"], row["PC2"]),
                        fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_title("Product Demand Segments (PCA view)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Sub-Categories by Cluster")
    display_cols = ["Sub-Category", "Cluster", "Demand Segment", "Total Sales",
                     "Growth Rate", "Sales Volatility", "Average Order Value"]
    st.dataframe(cluster_table[display_cols].sort_values("Cluster"))

    st.subheader("Recommended Stocking Strategy")
    for segment_name, group in cluster_table.groupby("Demand Segment"):
        with st.expander(f"{segment_name} ({len(group)} sub-categories)"):
            st.write(", ".join(group["Sub-Category"].tolist()))
            #working app link : https://b3aqqzdfgkfgxowacbnozn.streamlit.app
            
