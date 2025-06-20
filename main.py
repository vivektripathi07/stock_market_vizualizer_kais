import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(page_title="Trading Strategy Dashboard", layout="wide")
st.title("📈 Trading Strategy Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Parse datetime
    df['Entry time'] = pd.to_datetime(df['Entry time'])
    df['Month'] = pd.to_datetime(df['Month'])
    df['Trade Day'] = df['Entry time'].dt.date

    risk_free_annual = 0.005
    risk_free_daily = risk_free_annual / 252
    starting_balance = 100_000

    # ==== Cumulative Return ====
    df['Cumulative Return'] = df['Cumulative PNL'] / starting_balance
    st.subheader("📊 Cumulative Returns")
    fig = px.line(df, x='Entry time', y='Cumulative Return', title='Cumulative Return Over Time')
    fig.update_layout(yaxis_title="Return", xaxis_title="Date", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # ==== Monthly Returns ====
    monthly_returns = df.groupby(df['Month'].dt.to_period('M'))['PNL'].sum() / starting_balance
    monthly_returns = monthly_returns.to_timestamp()

    st.subheader("📅 Monthly Returns")
    fig = px.bar(
        monthly_returns, 
        x=monthly_returns.index, 
        y=monthly_returns.values, 
        labels={'x': 'Month', 'y': 'Return'},
        title="Monthly Return"
    )
    fig.update_layout(xaxis_title="Month", yaxis_title="Return", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # ==== Monthly Return Distribution ====
    st.subheader("📈 Distribution of Monthly Returns")
    fig = px.histogram(monthly_returns, nbins=10, histnorm='probability density', title="Histogram of Monthly Returns")
    fig.update_layout(xaxis_title="Return", yaxis_title="Density", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # ==== Daily Number of Trades ====
    daily_trades = df.groupby('Trade Day').size()
    st.subheader("📆 Daily Number of Trades")
    fig = px.line(
        x=daily_trades.index, 
        y=daily_trades.values, 
        labels={"x": "Date", "y": "Number of Trades"},
        title="Number of Trades Per Day"
    )
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # ==== Drawdown after Aug 2024 ====
    after_aug = df[df['Entry time'] >= '2024-08-01'].copy()
    after_aug['Cumulative Return'] = after_aug['Cumulative PNL'] / starting_balance
    after_aug['Running Max'] = after_aug['Cumulative Return'].cummax()
    after_aug['Drawdown'] = after_aug['Cumulative Return'] - after_aug['Running Max']
    st.subheader("📉 Drawdown After August 2024")
    fig = px.line(after_aug, x='Entry time', y='Drawdown', title="Drawdown Post August 2024")
    fig.update_layout(yaxis_title="Drawdown", xaxis_title="Date", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # ==== Average Daily Leverage ====
    avg_leverage = df.groupby('Trade Day')['Leverage'].mean()
    st.subheader("⚖️ Average Daily Leverage")
    fig = px.line(x=avg_leverage.index, y=avg_leverage.values, labels={"x": "Date", "y": "Leverage"}, title="Average Daily Leverage")
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # ==== Sharpe & Sortino ====
    returns_from_net_pct = df['Net profit (%)'] / 100
    returns_from_net_pct = returns_from_net_pct.dropna()
    mean_ret = returns_from_net_pct.mean()
    std_ret = returns_from_net_pct.std()
    sharpe = mean_ret / std_ret * np.sqrt(252)
    downside_std = returns_from_net_pct[returns_from_net_pct < 0].std()
    sortino = (mean_ret / downside_std * np.sqrt(252)) if downside_std != 0 else float('inf')

    st.subheader("📌 Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sharpe Ratio", f"{sharpe:.2f}")
    col2.metric("Sortino Ratio", f"{sortino:.2f}")
    col3.metric("Mean Return (%)", f"{mean_ret * 100:.2f}")
    col4.metric("Std Dev Return (%)", f"{std_ret * 100:.2f}")

    # ==== PNL Validation ====
    initial_portfolio = 100_000
    cumulative_pnl = 0
    results = []

    for _, trade in df.iterrows():
        side = trade["Side"]
        entry_price = trade["Avg. entry price"]
        exit_price = trade["Avg. exit price"]
        qty = trade["QTY"]
        commission = trade["Commission"]
        funding = trade["Funding"]
        leverage = trade["Leverage"]
        reported_pnl = trade["PNL"]

        if side == "LONG":
            calc_pnl = qty * (exit_price - entry_price)
        else:
            calc_pnl = qty * (entry_price - exit_price)

        calc_pnl_net = calc_pnl - commission + funding

        if side == "LONG":
            net_profit_pct = ((exit_price - entry_price) / entry_price) * 100 * leverage
        else:
            net_profit_pct = ((entry_price - exit_price) / entry_price) * 100 * leverage

        is_correct = abs(calc_pnl_net - reported_pnl) < 0.01
        cumulative_pnl += calc_pnl_net
        portfolio_value = initial_portfolio + cumulative_pnl

        results.append({
            "Entry Time": trade["Entry time"],
            "Ticker": trade["Ticker"],
            "Side": side,
            "Reported PNL": reported_pnl,
            "Calculated PNL": calc_pnl_net,
            "Validation": "✅" if is_correct else "❌",
            "Portfolio Value": portfolio_value
        })

    results_df = pd.DataFrame(results)

    # ==== Daily Portfolio Value ====
    st.subheader("💰 Portfolio Value Over Time")
    results_df["Entry Time"] = pd.to_datetime(results_df["Entry Time"])
    daily_pnl = results_df.set_index("Entry Time").resample("D")["Calculated PNL"].sum().reset_index()
    daily_pnl["Cumulative PNL"] = daily_pnl["Calculated PNL"].cumsum()
    daily_pnl["Portfolio Value"] = initial_portfolio + daily_pnl["Cumulative PNL"]

    fig = px.line(daily_pnl, x="Entry Time", y="Portfolio Value", title="Daily Portfolio Value")
    fig.add_hline(y=initial_portfolio, line_dash="dash", line_color="red",
                  annotation_text="Initial Capital", annotation_position="bottom right")
    fig.update_layout(hovermode="x unified", yaxis_tickprefix="$", yaxis_tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True)

    # ==== Summary ====
    st.subheader("📋 Summary")
    st.write(f"**Initial Portfolio:** ${initial_portfolio:,.2f}")
    st.write(f"**Final Portfolio:** ${portfolio_value:,.2f}")
    st.write(f"**Total PnL:** ${cumulative_pnl:,.2f} ({cumulative_pnl / initial_portfolio * 100:.2f}%)")

    # Expandable: Raw PNL Validation Table
    with st.expander("🔍 PNL Validation Table"):
        st.dataframe(results_df)

else:
    st.info("Please upload a CSV file to see the dashboard.")
