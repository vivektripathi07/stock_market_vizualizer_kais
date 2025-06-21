import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import skew, kurtosis

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

    # ==== Monthly Returns ====
    monthly_returns = df.groupby(df['Month'].dt.to_period('M'))['PNL'].sum() / starting_balance
    monthly_returns = monthly_returns.to_timestamp()

    # ==== Daily Number of Trades ====
    daily_trades = df.groupby('Trade Day').size()

    # ==== Drawdown after Aug 2024 ====
    after_aug = df[df['Entry time'] >= '2024-08-01'].copy()
    after_aug['Cumulative Return'] = after_aug['Cumulative PNL'] / starting_balance
    after_aug['Running Max'] = after_aug['Cumulative Return'].cummax()
    after_aug['Drawdown'] = after_aug['Cumulative Return'] - after_aug['Running Max']

    # Now calculate max drawdown and other metrics that rely on 'Drawdown'
    max_drawdown = after_aug['Drawdown'].min() * 100  # Convert to percentage

    # Calculate Longest Drawdown Days
    longest_dd_days = (after_aug[after_aug['Drawdown'] < 0].groupby((after_aug['Drawdown'] >= 0).cumsum()).size().max())

    # ==== Average Daily Leverage ====
    avg_leverage = df.groupby('Trade Day')['Leverage'].mean()

    # ==== Sharpe & Sortino ====
    returns = df['Net profit (%)'] / 100
    returns = returns.dropna()

    # Mean and standard deviation of returns
    mean_return = returns.mean()
    std_return = returns.std()

    # Daily Sharpe Ratio (assuming 252 trading days/year)
    if std_return != 0:
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252)
    else:
        sharpe_ratio = float('inf')

    # Downside deviation (only negative returns)
    downside_std = returns[returns < 0].std()

    # Daily Sortino Ratio
    if downside_std != 0:
        sortino_ratio = (mean_return / downside_std) * np.sqrt(252)
    else:
        sortino_ratio = float('inf')

    # PNL Validation
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
    daily_pnl = results_df.set_index("Entry Time").resample("D")["Calculated PNL"].sum().reset_index()
    daily_pnl["Cumulative PNL"] = daily_pnl["Calculated PNL"].cumsum()
    daily_pnl["Portfolio Value"] = initial_portfolio + daily_pnl["Cumulative PNL"]

    cumulative_return = (df['Cumulative PNL'].iloc[-1]) 

    # Calculate CAGR
    cagr = cumulative_return / 100

    # Calculate Volatility (annualized)
    daily_returns = df['Net profit (%)'] / 100  # Convert to decimal returns
    volatility = daily_returns.std() * np.sqrt(252) * 100  # Annualize the standard deviation

    # Calculate Calmar Ratio
    calmar_ratio = cagr / abs(max_drawdown / 100)

    # Calculate Skewness and Kurtosis
    skewness = skew(daily_returns.dropna())
    kurtosis_value = kurtosis(daily_returns.dropna())

    # Layout: Wide charts in center, metrics on the right
    left_col, right_col = st.columns([4, 1])  # adjust ratio as needed

    # ===== LEFT COLUMN =====
    with left_col:
        st.subheader("📊 Cumulative Returns")
        fig = px.line(df, x='Entry time', y='Cumulative Return', title='Cumulative Return Over Time')
        fig.update_layout(yaxis_title="Return", xaxis_title="Date", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

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

        st.subheader("📈 Distribution of Monthly Returns")
        fig = px.histogram(monthly_returns, nbins=10, histnorm='probability density', title="Histogram of Monthly Returns")
        fig.update_layout(xaxis_title="Return", yaxis_title="Density", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📆 Daily Number of Trades")
        fig = px.line(
            x=daily_trades.index, 
            y=daily_trades.values, 
            labels={"x": "Date", "y": "Number of Trades"},
            title="Number of Trades Per Day"
        )
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📉 Drawdown After August 2024")
        fig = px.line(after_aug, x='Entry time', y='Drawdown', title="Drawdown Post August 2024")
        fig.update_layout(yaxis_title="Drawdown", xaxis_title="Date", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("⚖️ Average Daily Leverage")
        fig = px.line(x=avg_leverage.index, y=avg_leverage.values, labels={"x": "Date", "y": "Leverage"}, title="Average Daily Leverage")
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("💰 Portfolio Value Over Time")
        fig = px.line(daily_pnl, x="Entry Time", y="Portfolio Value", title="Daily Portfolio Value")
        fig.add_hline(y=initial_portfolio, line_dash="dash", line_color="red",
                      annotation_text="Initial Capital", annotation_position="bottom right")
        fig.update_layout(hovermode="x unified", yaxis_tickprefix="$", yaxis_tickformat=",.0f")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("🔍 PNL Validation Table"):
            st.dataframe(results_df)

    # ===== RIGHT COLUMN =====
    with right_col:
        # CSS to reduce font sizes across all elements
        st.markdown("""
        <style>
        /* Target Subheaders (h3) */
        h3 {
            font-size: 16px !important;
            margin-bottom: 0.5rem !important;
        }
        
        /* Target Metric Titles and Values */
        div[data-testid="stMetricLabel"] > div, 
        div[data-testid="stMetricValue"] > div {
            font-size: 14px !important;
        }
        
        /* Target Regular Text (st.write) */
        .stMarkdown p {
            font-size: 14px !important;
            margin-bottom: 0.5rem !important;
        }
        
        /* Optional: Reduce space between metrics */
        div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column"] > div {
            padding-top: 0px !important;
        }
        </style>
        """, unsafe_allow_html=True)

        # --- Your Original Content ---
        st.subheader("📌 Performance Metrics")  # Now smaller (h3 target)
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")  # Title and value shrunk
        st.metric("Sortino Ratio", f"{sortino_ratio:.2f}")
        st.metric("Mean Return (%)", f"{mean_return * 100:.2f}")
        st.metric("Std Dev Return (%)", f"{std_return * 100:.2f}")
        
        # New metrics
        st.metric("Cumulative Return (%)", f"{cumulative_return:.2f}")
        st.metric("CAGR (%)", f"{cagr:.2f}")
        st.metric("Max Drawdown (%)", f"{max_drawdown:.2f}")
        st.metric("Longest DD Days", f"{longest_dd_days}")
        st.metric("Volatility (ann.)", f"{volatility:.2f}")
        st.metric("Calmar Ratio", f"{calmar_ratio:.2f}")
        st.metric("Skew", f"{skewness:.2f}")
        st.metric("Kurtosis", f"{kurtosis_value:.2f}")
        
        st.subheader("📋 Summary")  # Also smaller
        st.write(f"**Initial Portfolio:** ${initial_portfolio:,.2f}")  # Smaller text
        st.write(f"**Final Portfolio:** ${portfolio_value:,.2f}")
        st.write(f"**Total PnL:** ${cumulative_pnl:,.2f} ({cumulative_pnl / initial_portfolio * 100:.2f}%)")

else:
    st.info("Please upload a CSV file to see the dashboard.")
