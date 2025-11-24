import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Streamlit page setup
# -----------------------------
st.set_page_config(page_title="Commodity Risk Engine", layout="wide")
st.title("Commodity Risk Engine")
st.caption("Monte Carlo VaR & ES with Price, Basis, FX & Freight risk")

# -----------------------------
# Helper math utilities
# -----------------------------
def cholesky_psd(matrix, jitter_start=1e-10, max_tries=7):
    """Cholesky with jitter for near-PSD matrices."""
    A = np.array(matrix, dtype=float)
    for i in range(max_tries):
        try:
            return np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            jitter = jitter_start * (10 ** i)
            A = A + np.eye(A.shape[0]) * jitter
    # Last resort: eigenvalue clip
    w, v = np.linalg.eigh(A)
    w = np.clip(w, 1e-12, None)
    A_psd = (v * w) @ v.T
    return np.linalg.cholesky(A_psd)

def random_student_t_normals(n_dim, df):
    """Generate Student-t via normal scaled by chi-square."""
    z = np.random.normal(size=n_dim)
    chi = np.random.chisquare(df)
    scale = np.sqrt(df / chi)
    return z * scale

def format_currency(x, short=False):
    sign = "-" if x < 0 else ""
    ax = abs(x)
    if short and ax >= 1_000_000:
        return f"{sign}${ax/1_000_000:,.2f}M"
    return f"{sign}${ax:,.0f}"

# -----------------------------
# Sidebar / Global defaults
# -----------------------------
with st.sidebar:
    st.header("Global Settings")
    st.markdown("Use the tabs to configure the portfolio, risk factors, run the simulation and analyze results.")

# -----------------------------
# Tabs
# -----------------------------
tabs = st.tabs([
    "Portfolio",
    "Risk Factors",
    "Simulation",
    "Results",
    "Hedging",
    "Stress Testing",
    "Risk Decomposition"
])

# -----------------------------
# Tab: Portfolio
# -----------------------------
with tabs[0]:
    st.header("Portfolio Configuration")

    col1, col2, col3, col4 = st.columns([1,1,1,1])
    # Quantities and prices
    corn_qty   = col1.number_input("Corn Quantity (bu)", min_value=0, step=1000, value=1_000_000)
    corn_price = col1.number_input("Corn Price ($/bu)", min_value=0.0, step=0.01, value=4.50, format="%.2f")

    soy_qty    = col2.number_input("Soybeans Quantity (bu)", min_value=0, step=1000, value=600_000)
    soy_price  = col2.number_input("Soybeans Price ($/bu)", min_value=0.0, step=0.01, value=11.50, format="%.2f")

    wheat_qty  = col3.number_input("Wheat Quantity (bu)", min_value=0, step=1000, value=800_000)
    wheat_price= col3.number_input("Wheat Price ($/bu)", min_value=0.0, step=0.01, value=5.50, format="%.2f")

    contract_corn   = col4.number_input("Corn Contract Size", min_value=1, value=5000, step=1)
    contract_soy    = col4.number_input("Soybeans Contract Size", min_value=1, value=5000, step=1)
    contract_wheat  = col4.number_input("Wheat Contract Size", min_value=1, value=5000, step=1)

    corn_value  = corn_qty * corn_price
    soy_value   = soy_qty  * soy_price
    wheat_value = wheat_qty* wheat_price
    total_value = corn_value + soy_value + wheat_value

    st.metric("Total Portfolio Value", format_currency(total_value))

    c1, c2, c3 = st.columns(3)
    c1.metric("Corn Exposure",  format_currency(corn_value, short=True), f"{(corn_value/total_value*100 if total_value>0 else 0):.1f}%")
    c2.metric("Soybeans Exposure", format_currency(soy_value, short=True), f"{(soy_value/total_value*100 if total_value>0 else 0):.1f}%")
    c3.metric("Wheat Exposure", format_currency(wheat_value, short=True), f"{(wheat_value/total_value*100 if total_value>0 else 0):.1f}%")

    # Simple composition chart
    fig, ax = plt.subplots()
    labels = ["Corn", "Soybeans", "Wheat"]
    sizes = [corn_value, soy_value, wheat_value]
    if sum(sizes) == 0:
        sizes = [1,1,1]
    ax.pie(sizes, labels=labels, autopct="%1.1f%%")
    ax.set_title("Portfolio Composition")
    st.pyplot(fig)

# -----------------------------
# Tab: Risk Factors
# -----------------------------
with tabs[1]:
    st.header("Risk Factors Configuration")

    # Toggles (mirroring the JS behavior where price is always considered in returns)
    tcol1, tcol2, tcol3, tcol4 = st.columns(4)
    price_enabled   = tcol1.checkbox("Enable Price Risk", value=True)
    basis_enabled   = tcol2.checkbox("Enable Basis Risk", value=True)
    fx_enabled      = tcol3.checkbox("Enable FX Risk", value=True)
    freight_enabled = tcol4.checkbox("Enable Freight Risk", value=True)

    # Price risk
    st.subheader("Price Risk")
    pcol1, pcol2, pcol3 = st.columns(3)
    price_vol = pcol1.number_input("Annualized Price Vol (%)", min_value=0.0, max_value=500.0, value=18.0, step=0.1)
    corr_cs  = pcol2.number_input("Corr: Corn-Soy",   min_value=-1.0, max_value=1.0, value=0.47, step=0.01)
    corr_cw  = pcol2.number_input("Corr: Corn-Wheat", min_value=-1.0, max_value=1.0, value=0.37, step=0.01)
    corr_sw  = pcol2.number_input("Corr: Soy-Wheat",  min_value=-1.0, max_value=1.0, value=0.41, step=0.01)

    # Basis risk
    st.subheader("Basis Risk (Cash - Futures)")
    bcol1, bcol2, bcol3, bcol4 = st.columns(4)
    basis_vol = bcol1.number_input("Basis Vol (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
    basis_corr = bcol2.number_input("Correlation with Price", min_value=-1.0, max_value=1.0, value=0.3, step=0.01)
    basis_mean_corn = bcol3.number_input("Mean Corn Basis (%)", value=-5.0, step=0.1, help="as % of price")
    basis_mean_soy  = bcol3.number_input("Mean Soy Basis (%)", value=-8.0, step=0.1, help="as % of price")
    basis_mean_wht  = bcol3.number_input("Mean Wheat Basis (%)", value=-6.0, step=0.1, help="as % of price")

    # FX risk
    st.subheader("FX Risk")
    fxc1, fxc2, fxc3, fxc4 = st.columns(4)
    fx_pair = fxc1.selectbox("Currency Pair", ["USD/BRL", "USD/CAD", "USD/INR", "EUR/USD"], index=0)
    default_rates = {
        "USD/BRL": {"rate": 5.20, "vol": 15.0, "corr": -0.20},
        "USD/CAD": {"rate": 1.35, "vol": 8.0, "corr": -0.15},
        "USD/INR": {"rate": 83.50, "vol": 6.0, "corr": 0.05},
        "EUR/USD": {"rate": 1.08, "vol": 10.0, "corr": -0.10},
    }
    _d = default_rates[fx_pair]
    fx_rate = fxc2.number_input("Current FX Rate", value=_d["rate"], step=0.01, format="%.2f")
    fx_vol  = fxc3.number_input("FX Vol (%)", value=_d["vol"], step=0.1)
    fx_corr = fxc4.number_input("FX Corr with Prices", min_value=-1.0, max_value=1.0, value=_d["corr"], step=0.01)

    # Freight risk
    st.subheader("Freight Risk")
    frc1, frc2, frc3 = st.columns(3)
    freight_cost = frc1.number_input("Freight Cost ($/bu)", value=0.50, step=0.01, format="%.2f")
    freight_vol  = frc2.number_input("Freight Vol (%)", value=35.0, step=0.1)
    freight_corr = frc3.number_input("Freight Corr with Prices", min_value=-1.0, max_value=1.0, value=0.25, step=0.01)

    # Risk factor status
    st.info(
        f"Status — Price: {'Enabled' if price_enabled else 'Disabled'} | "
        f"Basis: {'Enabled' if basis_enabled else 'Disabled'} | "
        f"FX: {'Enabled' if fx_enabled else 'Disabled'} | "
        f"Freight: {'Enabled' if freight_enabled else 'Disabled'}"
    )

# -----------------------------
# Tab: Simulation
# -----------------------------
with tabs[2]:
    st.header("Monte Carlo Simulation")
    scol1, scol2, scol3, scol4 = st.columns(4)
    num_scenarios = scol1.number_input("Number of Scenarios", min_value=1000, max_value=200_000, step=1000, value=25_000)
    horizon = scol2.number_input("Time Horizon (days)", min_value=1, max_value=365, step=1, value=20)
    distribution = scol3.selectbox("Distribution", ["Student-t (Fat Tails)", "Normal"], index=0)
    df = scol4.number_input("Degrees of Freedom (t)", min_value=3, max_value=100, step=1, value=6)

    run = st.button("Run Simulation", use_container_width=True)

    if run:
        # Daily vols
        price_vol_d = (price_vol / 100.0) / np.sqrt(252.0)
        basis_vol_d = (basis_vol / 100.0) / np.sqrt(252.0)
        fx_vol_d    = (fx_vol / 100.0) / np.sqrt(252.0)
        freight_vol_d = (freight_vol / 100.0) / np.sqrt(252.0)

        # Scale by horizon
        s_price   = price_vol_d * np.sqrt(horizon)
        s_basis   = basis_vol_d * np.sqrt(horizon)
        s_fx      = fx_vol_d * np.sqrt(horizon)
        s_freight = freight_vol_d * np.sqrt(horizon)

        # Correlation matrix (8 x 8)
        # Order: Corn P, Soy P, Wheat P, Corn Basis, Soy Basis, Wheat Basis, FX, Freight
        corr_matrix = np.array([
            [1,      corr_cs, corr_cw, basis_corr, 0,          0,          fx_corr,     freight_corr],
            [corr_cs,1,       corr_sw, 0,          basis_corr, 0,          fx_corr,     freight_corr],
            [corr_cw,corr_sw, 1,       0,          0,          basis_corr, fx_corr,     freight_corr],
            [basis_corr,0,    0,       1,          0,          0,          0,           0],
            [0,      basis_corr,0,     0,          1,          0,          0,           0],
            [0,      0,      basis_corr,0,         0,          1,          0,           0],
            [fx_corr,fx_corr, fx_corr, 0,          0,          0,          1,           0],
            [freight_corr,freight_corr,freight_corr,0,0,        0,          0,           1]
        ], dtype=float)

        L = cholesky_psd(corr_matrix)

        # Portfolio dicts
        prices = {"corn": corn_price, "soybeans": soy_price, "wheat": wheat_price}
        qtys   = {"corn": corn_qty,   "soybeans": soy_qty,   "wheat": wheat_qty}
        basis_means = {"corn": basis_mean_corn/100.0, "soybeans": basis_mean_soy/100.0, "wheat": basis_mean_wht/100.0}

        # Simulate
        scen_pnls = np.zeros(num_scenarios)
        price_ret_store = np.zeros((num_scenarios,3))
        basis_ret_store = np.zeros((num_scenarios,3))
        fx_ret_store    = np.zeros(num_scenarios)
        freight_ret_store = np.zeros(num_scenarios)

        for i in range(num_scenarios):
            # independent shocks
            if distribution.startswith("Student"):
                z = random_student_t_normals(8, int(df))
            else:
                z = np.random.normal(size=8)
            # correlate
            shocks = L @ z

            # scale
            price_returns = {
                "corn":   shocks[0] * s_price,
                "soybeans": shocks[1] * s_price,
                "wheat":  shocks[2] * s_price,
            }
            basis_returns = {
                "corn":   shocks[3] * s_basis,
                "soybeans": shocks[4] * s_basis,
                "wheat":  shocks[5] * s_basis,
            }
            fx_return = shocks[6] * s_fx
            freight_return = shocks[7] * s_freight

            # compute scenario PnL
            total_pnl = 0.0
            for k in ["corn","soybeans","wheat"]:
                price0 = prices[k]
                new_price = price0 * (1.0 + price_returns[k])
                if basis_enabled:
                    new_price += price0 * (basis_means[k] + basis_returns[k])
                if fx_enabled:
                    new_fx = fx_rate * (1.0 + fx_return)
                    new_price = new_price * new_fx / fx_rate
                if freight_enabled:
                    new_price -= (freight_cost * (1.0 + freight_return))
                # PnL
                total_pnl += qtys[k] * (new_price - price0)

            scen_pnls[i] = total_pnl
            price_ret_store[i,:] = [price_returns["corn"], price_returns["soybeans"], price_returns["wheat"]]
            basis_ret_store[i,:] = [basis_returns["corn"], basis_returns["soybeans"], basis_returns["wheat"]]
            fx_ret_store[i] = fx_return
            freight_ret_store[i] = freight_return

        # Sort for quantiles
        order = np.argsort(scen_pnls)
        pnls_sorted = scen_pnls[order]

        var95 = -pnls_sorted[int(0.05 * num_scenarios)]
        var99 = -pnls_sorted[int(0.01 * num_scenarios)]
        es975 = -pnls_sorted[:int(0.025 * num_scenarios)].mean()

        mean_pnl = scen_pnls.mean()
        median_pnl = np.median(scen_pnls)
        std_pnl = scen_pnls.std()

        # Risk contributions (approximate variance decomposition)
        price_component = np.sum(np.abs(price_ret_store), axis=1)  # sum of abs per scenario (3 commodities)
        basis_component = np.sum(np.abs(basis_ret_store), axis=1)
        fx_component    = np.abs(fx_ret_store)
        freight_component = np.abs(freight_ret_store)

        price_var = np.sum(price_component**2)
        basis_var = np.sum(basis_component**2)
        fx_var    = np.sum(fx_component**2)
        freight_var = np.sum(freight_component**2)
        total_var_proxy = price_var + basis_var + fx_var + freight_var + 1e-12

        contrib = {
            "price":   100.0 * price_var / total_var_proxy,
            "basis":   100.0 * basis_var / total_var_proxy,
            "fx":      100.0 * fx_var / total_var_proxy,
            "freight": 100.0 * freight_var / total_var_proxy,
        }

        st.session_state["results"] = dict(
            scen_pnls=scen_pnls,
            var95=var95, var99=var99, es975=es975,
            mean=mean_pnl, median=median_pnl, std=std_pnl,
            contrib=contrib, horizon=horizon,
            prices=prices, qtys=qtys,
            contract_sizes={"corn": contract_corn, "soybeans": contract_soy, "wheat": contract_wheat},
        )
        st.success("Simulation complete. See the Results / Hedging / Decomposition tabs.")

# -----------------------------
# Tab: Results
# -----------------------------
with tabs[3]:
    st.header("Results Dashboard")
    if "results" in st.session_state:
        r = st.session_state["results"]

        c = st.columns(6)
        c[0].metric(f"95% VaR ({r['horizon']}d)", format_currency(r["var95"]))
        c[1].metric(f"99% VaR ({r['horizon']}d)", format_currency(r["var99"]))
        c[2].metric("Expected Shortfall (97.5%)", format_currency(r["es975"]))
        c[3].metric("Mean P&L", format_currency(r["mean"]))
        c[4].metric("Median P&L", format_currency(r["median"]))
        c[5].metric("Std Dev", format_currency(abs(r["std"])))

        st.subheader("Risk Contribution by Factor")
        k = ["price","basis","fx","freight"]
        vals = [r["contrib"][x] for x in k]
        df = pd.DataFrame({"Factor": [x.capitalize() for x in k], "Contribution (%)": vals})
        st.dataframe(df, hide_index=True)

        # Histogram
        fig, ax = plt.subplots()
        ax.hist(r["scen_pnls"], bins=50)
        ax.set_title("P&L Distribution")
        ax.set_xlabel("P&L ($)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    else:
        st.info("Run a simulation first.")

# -----------------------------
# Tab: Hedging
# -----------------------------
with tabs[4]:
    st.header("Hedging Optimization")
    if "results" in st.session_state:
        r = st.session_state["results"]
        hedge_ratio = 0.80  # 80%
        eff = 0.85         # 85% effectiveness

        contracts = {
            "corn":   int(round((r["qtys"]["corn"]   * hedge_ratio) / r["contract_sizes"]["corn"])),
            "soybeans": int(round((r["qtys"]["soybeans"] * hedge_ratio) / r["contract_sizes"]["soybeans"])),
            "wheat":  int(round((r["qtys"]["wheat"]  * hedge_ratio) / r["contract_sizes"]["wheat"])),
        }

        hedged_var95 = r["var95"] * (1 - hedge_ratio * eff)
        hedged_var99 = r["var99"] * (1 - hedge_ratio * eff)
        red_pct = 100.0 * (r["var95"] - hedged_var95) / (r["var95"] + 1e-12)

        cols = st.columns(4)
        cols[0].metric("Corn contracts (short)", f"{contracts['corn']}")
        cols[1].metric("Soybeans contracts (short)", f"{contracts['soybeans']}")
        cols[2].metric("Wheat contracts (short)", f"{contracts['wheat']}")
        cols[3].metric("Expected VaR Reduction", f"{red_pct:.1f}%")

        lcol, rcol = st.columns(2)
        with lcol:
            st.subheader("Unhedged")
            st.metric("95% VaR", format_currency(r["var95"]))
            st.metric("99% VaR", format_currency(r["var99"]))
        with rcol:
            st.subheader("Hedged (80%, 85% eff.)")
            st.metric("95% VaR", format_currency(hedged_var95), f"↓ {(r['var95']-hedged_var95)/1_000_000:.2f}M")
            st.metric("99% VaR", format_currency(hedged_var99), f"↓ {(r['var99']-hedged_var99)/1_000_000:.2f}M")
    else:
        st.info("Run a simulation first.")

# -----------------------------
# Tab: Stress Testing
# -----------------------------
with tabs[5]:
    st.header("Stress Testing")
    def stress_pnl(price_shock, basis_shock, fx_shock, freight_shock):
        total = 0.0
        for k, price0 in {"corn": corn_price, "soybeans": soy_price, "wheat": wheat_price}.items():
            new_price = price0 * (1.0 + price_shock)
            if basis_enabled:
                # use mean basis in percent terms
                m = {"corn": basis_mean_corn/100.0, "soybeans": basis_mean_soy/100.0, "wheat": basis_mean_wht/100.0}[k]
                new_price += price0 * (m + basis_shock)
            if fx_enabled:
                new_fx = fx_rate * (1.0 + fx_shock)
                new_price = new_price * new_fx / fx_rate
            if freight_enabled:
                new_price -= (freight_cost * (1.0 + freight_shock))
            q = {"corn": corn_qty, "soybeans": soy_qty, "wheat": wheat_qty}[k]
            total += q * (new_price - price0)
        return total

    c1, c2, c3 = st.columns(3)
    if c1.button("Moderate Downturn"):
        st.session_state["stress_md"] = stress_pnl(-0.20, +0.10, +0.05, +0.15)
    if c2.button("Severe Crisis"):
        st.session_state["stress_sc"] = stress_pnl(-0.40, +0.20, +0.10, +0.30)
    if c3.button("Perfect Storm"):
        st.session_state["stress_ps"] = stress_pnl(-0.30, +0.25, +0.15, +0.40)

    st.markdown("#### Custom Scenario")
    s1, s2, s3, s4 = st.columns(4)
    custom_price = s1.slider("Price Shock (%)", -50, 50, 0) / 100.0
    custom_basis = s2.slider("Basis Shock (%)", -50, 50, 0) / 100.0
    custom_fx    = s3.slider("FX Shock (%)", -50, 50, 0) / 100.0
    custom_freight = s4.slider("Freight Shock (%)", -50, 50, 0) / 100.0
    if st.button("Run Custom Stress", use_container_width=True):
        st.session_state["stress_custom"] = stress_pnl(custom_price, custom_basis, custom_fx, custom_freight)

    # Show results if present
    rows = []
    if "stress_md" in st.session_state: rows.append(("Moderate Downturn", st.session_state["stress_md"]))
    if "stress_sc" in st.session_state: rows.append(("Severe Crisis", st.session_state["stress_sc"]))
    if "stress_ps" in st.session_state: rows.append(("Perfect Storm", st.session_state["stress_ps"]))
    if "stress_custom" in st.session_state: rows.append(("Custom", st.session_state["stress_custom"]))

    if rows:
        df = pd.DataFrame(rows, columns=["Scenario", "P&L ($)"])
        st.dataframe(df.style.format({"P&L ($)": "{:,.0f}"}), hide_index=True)

        # Bar chart
        fig, ax = plt.subplots()
        ax.bar(df["Scenario"], df["P&L ($)"])
        ax.set_ylabel("P&L ($)")
        ax.set_title("Stress Test Results")
        st.pyplot(fig)

# -----------------------------
# Tab: Risk Decomposition
# -----------------------------
with tabs[6]:
    st.header("Risk Decomposition")
    if "results" in st.session_state:
        r = st.session_state["results"]
        total_value = (
            r["qtys"]["corn"] * r["prices"]["corn"] +
            r["qtys"]["soybeans"] * r["prices"]["soybeans"] +
            r["qtys"]["wheat"] * r["prices"]["wheat"]
        )
        w_c = (r["qtys"]["corn"] * r["prices"]["corn"]) / (total_value + 1e-12)
        w_s = (r["qtys"]["soybeans"] * r["prices"]["soybeans"]) / (total_value + 1e-12)
        w_w = (r["qtys"]["wheat"] * r["prices"]["wheat"]) / (total_value + 1e-12)

        # Simplified marginal VaR proportional to exposure * portfolio VaR * price vol (relative scale)
        # Here we use same proportionality factor for all three; goal is to mirror the JS simplification
        mv_c = w_c * r["var95"] * 0.18  # 18% base as in defaults
        mv_s = w_s * r["var95"] * 0.18
        mv_w = w_w * r["var95"] * 0.18

        cols = st.columns(3)
        cols[0].metric("Corn Marginal VaR", format_currency(mv_c), f"{w_c*100:.1f}% of portfolio")
        cols[1].metric("Soybeans Marginal VaR", format_currency(mv_s), f"{w_s*100:.1f}% of portfolio")
        cols[2].metric("Wheat Marginal VaR", format_currency(mv_w), f"{w_w*100:.1f}% of portfolio")

        st.subheader("Risk Factor Analysis")
        table = pd.DataFrame({
            "Factor": ["Price Risk", "Basis Risk", "FX Risk", "Freight Risk"],
            "Volatility (input)": ["—", "—", "—", "—"],
            "Contribution to VaR (%)": [r["contrib"]["price"], r["contrib"]["basis"], r["contrib"]["fx"], r["contrib"]["freight"]],
            "Status": ["Active"]*4
        })
        st.dataframe(table.style.format({"Contribution to VaR (%)": "{:.1f}"}), hide_index=True)

        es_to_var = (r["es975"] / (r["var95"] + 1e-12))
        st.info(f"Tail Risk: ES/VaR = {es_to_var:.2f} → {'significant' if es_to_var > 1.3 else 'moderate'} tail risk")
    else:
        st.info("Run a simulation first.")
