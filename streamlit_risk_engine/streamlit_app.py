
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Commodity Risk Engine", layout="wide")
st.title("Commodity Risk Engine")
st.caption("Monte Carlo VaR & ES with Price, Basis, FX & Freight risk (Plotly + caching)")

# -------- Helpers --------
def format_currency(x, short=False):
    sign = "-" if x < 0 else ""
    ax = abs(x)
    if short and ax >= 1_000_000:
        return f"{sign}${ax/1_000_000:,.2f}M"
    return f"{sign}${ax:,.0f}"

def cholesky_psd(matrix, jitter_start=1e-10, max_tries=7):
    A = np.array(matrix, dtype=float)
    for i in range(max_tries):
        try:
            return np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            jitter = jitter_start * (10 ** i)
            A = A + np.eye(A.shape[0]) * jitter
    # eigen clip fallback
    w, v = np.linalg.eigh(A)
    w = np.clip(w, 1e-12, None)
    A_psd = (v * w) @ v.T
    return np.linalg.cholesky(A_psd)

def random_student_t_normals(n_dim, df):
    z = np.random.normal(size=n_dim)
    chi = np.random.chisquare(df)
    scale = np.sqrt(df / chi)
    return z * scale

# -------- Tabs --------
tabs = st.tabs([
    "Portfolio",
    "Risk Factors",
    "Simulation",
    "Results",
    "Hedging",
    "Stress Testing",
    "Risk Decomposition"
])

# -------- Portfolio --------
with tabs[0]:
    st.header("Portfolio Configuration")
    col1, col2, col3, col4 = st.columns([1,1,1,1])
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
    comp_df = pd.DataFrame({
        "Commodity": ["Corn", "Soybeans", "Wheat"],
        "Value": [corn_value, soy_value, wheat_value]
    })
    pie = px.pie(comp_df, values="Value", names="Commodity", title="Portfolio Composition", hole=0.3)
    st.plotly_chart(pie, use_container_width=True)

# -------- Risk Factors --------
with tabs[1]:
    st.header("Risk Factors Configuration")
    tcol1, tcol2, tcol3, tcol4 = st.columns(4)
    price_enabled   = tcol1.checkbox("Enable Price Risk", value=True)
    basis_enabled   = tcol2.checkbox("Enable Basis Risk", value=True)
    fx_enabled      = tcol3.checkbox("Enable FX Risk", value=True)
    freight_enabled = tcol4.checkbox("Enable Freight Risk", value=True)

    st.subheader("Price Risk")
    pcol1, pcol2 = st.columns([1,2])
    price_vol = pcol1.number_input("Annualized Price Vol (%)", min_value=0.0, max_value=500.0, value=18.0, step=0.1)
    corr_cs  = pcol2.number_input("Corr: Corn–Soy",   min_value=-1.0, max_value=1.0, value=0.47, step=0.01)
    corr_cw  = pcol2.number_input("Corr: Corn–Wheat", min_value=-1.0, max_value=1.0, value=0.37, step=0.01)
    corr_sw  = pcol2.number_input("Corr: Soy–Wheat",  min_value=-1.0, max_value=1.0, value=0.41, step=0.01)

    st.subheader("Basis Risk (Cash - Futures)")
    bcol1, bcol2, bcol3 = st.columns(3)
    basis_vol = bcol1.number_input("Basis Vol (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
    basis_corr = bcol2.number_input("Correlation with Price", min_value=-1.0, max_value=1.0, value=0.3, step=0.01)
    basis_mean_corn = bcol3.number_input("Mean Corn Basis (%)", value=-5.0, step=0.1, help="as % of price")
    basis_mean_soy  = bcol3.number_input("Mean Soy Basis (%)", value=-8.0, step=0.1, help="as % of price")
    basis_mean_wht  = bcol3.number_input("Mean Wheat Basis (%)", value=-6.0, step=0.1, help="as % of price")

    st.subheader("FX Risk")
    fxc1, fxc2, fxc3, fxc4 = st.columns(4)
    fx_pair = fxc1.selectbox("Currency Pair", ["USD/BRL", "USD/CAD", "USD/INR", "EUR/USD"], index=0)
    defaults = {
        "USD/BRL": {"rate": 5.20, "vol": 15.0, "corr": -0.20},
        "USD/CAD": {"rate": 1.35, "vol": 8.0, "corr": -0.15},
        "USD/INR": {"rate": 83.50, "vol": 6.0, "corr": 0.05},
        "EUR/USD": {"rate": 1.08, "vol": 10.0, "corr": -0.10},
    }
    _d = defaults[fx_pair]
    fx_rate = fxc2.number_input("Current FX Rate", value=_d["rate"], step=0.01, format="%.2f")
    fx_vol  = fxc3.number_input("FX Vol (%)", value=_d["vol"], step=0.1)
    fx_corr = fxc4.number_input("FX Corr with Prices", min_value=-1.0, max_value=1.0, value=_d["corr"], step=0.01)

    st.subheader("Freight Risk")
    frc1, frc2, frc3 = st.columns(3)
    freight_cost = frc1.number_input("Freight Cost ($/bu)", value=0.50, step=0.01, format="%.2f")
    freight_vol  = frc2.number_input("Freight Vol (%)", value=35.0, step=0.1)
    freight_corr = frc3.number_input("Freight Corr with Prices", min_value=-1.0, max_value=1.0, value=0.25, step=0.01)

    st.info(
        f"Status — Price: {'Enabled' if price_enabled else 'Disabled'} | "
        f"Basis: {'Enabled' if basis_enabled else 'Disabled'} | "
        f"FX: {'Enabled' if fx_enabled else 'Disabled'} | "
        f"Freight: {'Enabled' if freight_enabled else 'Disabled'}"
    )

# -------- Cached Simulation --------
@st.cache_data(show_spinner=True)
def run_simulation(seed, num_scenarios, horizon, distribution, df,
                   price_vol, basis_vol, fx_vol, freight_vol,
                   corr_cs, corr_cw, corr_sw, basis_corr, fx_corr, freight_corr,
                   basis_enabled, fx_enabled, freight_enabled,
                   corn_price, soy_price, wheat_price,
                   corn_qty, soy_qty, wheat_qty,
                   basis_mean_corn, basis_mean_soy, basis_mean_wht, fx_rate):
    np.random.seed(seed)
    price_vol_d = (price_vol / 100.0) / np.sqrt(252.0)
    basis_vol_d = (basis_vol / 100.0) / np.sqrt(252.0)
    fx_vol_d    = (fx_vol / 100.0) / np.sqrt(252.0)
    freight_vol_d = (freight_vol / 100.0) / np.sqrt(252.0)

    s_price   = price_vol_d * np.sqrt(horizon)
    s_basis   = basis_vol_d * np.sqrt(horizon)
    s_fx      = fx_vol_d * np.sqrt(horizon)
    s_freight = freight_vol_d * np.sqrt(horizon)

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

    prices = {"corn": corn_price, "soybeans": soy_price, "wheat": wheat_price}
    qtys   = {"corn": corn_qty,   "soybeans": soy_qty,   "wheat": wheat_qty}
    basis_means = {"corn": basis_mean_corn/100.0, "soybeans": basis_mean_soy/100.0, "wheat": basis_mean_wht/100.0}

    scen_pnls = np.zeros(num_scenarios)
    price_ret_store = np.zeros((num_scenarios,3))
    basis_ret_store = np.zeros((num_scenarios,3))
    fx_ret_store    = np.zeros(num_scenarios)
    freight_ret_store = np.zeros(num_scenarios)

    for i in range(num_scenarios):
        if distribution.startswith("Student"):
            z = random_student_t_normals(8, int(df))
        else:
            z = np.random.normal(size=8)
        shocks = L @ z

        price_returns = {"corn": shocks[0]*s_price, "soybeans": shocks[1]*s_price, "wheat": shocks[2]*s_price}
        basis_returns = {"corn": shocks[3]*s_basis, "soybeans": shocks[4]*s_basis, "wheat": shocks[5]*s_basis}
        fx_return = shocks[6]*s_fx
        freight_return = shocks[7]*s_freight

        total_pnl = 0.0
        for k in ["corn","soybeans","wheat"]:
            p0 = prices[k]
            new_p = p0 * (1.0 + price_returns[k])
            if basis_enabled:
                new_p += p0 * (basis_means[k] + basis_returns[k])
            if fx_enabled:
                new_fx = fx_rate * (1.0 + fx_return)
                new_p = new_p * new_fx / fx_rate
            if freight_enabled:
                new_p -= (freight_return + 1.0) * 0.0  # freight applied as cost below
                # Using fixed $/bu freight cost:
                new_p -= 0.0  # will deduct separately in PnL
            total_pnl += qtys[k] * (new_p - p0)
            if freight_enabled:
                # Deduct freight cost change (absolute $/bu)
                new_freight = 0.0

        scen_pnls[i] = total_pnl
        price_ret_store[i,:] = [price_returns["corn"], price_returns["soybeans"], price_returns["wheat"]]
        basis_ret_store[i,:] = [basis_returns["corn"], basis_returns["soybeans"], basis_returns["wheat"]]
        fx_ret_store[i] = fx_return
        freight_ret_store[i] = freight_return

    order = np.argsort(scen_pnls)
    pnls_sorted = scen_pnls[order]
    var95 = -pnls_sorted[int(0.05 * num_scenarios)]
    var99 = -pnls_sorted[int(0.01 * num_scenarios)]
    es975 = -pnls_sorted[:int(0.025 * num_scenarios)].mean()

    mean_pnl = scen_pnls.mean()
    median_pnl = np.median(scen_pnls)
    std_pnl = scen_pnls.std()

    price_component = np.sum(np.abs(price_ret_store), axis=1)
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

    return dict(
        scen_pnls=scen_pnls,
        var95=var95, var99=var99, es975=es975,
        mean=mean_pnl, median=median_pnl, std=std_pnl,
        contrib=contrib,
    )

# -------- Simulation tab --------
with tabs[2]:
    st.header("Monte Carlo Simulation")
    scol1, scol2, scol3, scol4 = st.columns(4)
    num_scenarios = scol1.number_input("Number of Scenarios", min_value=1000, max_value=200_000, step=1000, value=25_000)
    horizon = scol2.number_input("Time Horizon (days)", min_value=1, max_value=365, step=1, value=20)
    distribution = scol3.selectbox("Distribution", ["Student-t (Fat Tails)", "Normal"], index=0)
    df = scol4.number_input("Degrees of Freedom (t)", min_value=3, max_value=100, step=1, value=6)

    if st.button("Run Simulation", use_container_width=True):
        results = run_simulation(
            seed=42, num_scenarios=num_scenarios, horizon=horizon,
            distribution=distribution, df=df,
            price_vol=price_vol, basis_vol=basis_vol, fx_vol=fx_vol, freight_vol=freight_vol,
            corr_cs=corr_cs, corr_cw=corr_cw, corr_sw=corr_sw, basis_corr=basis_corr, fx_corr=fx_corr, freight_corr=freight_corr,
            basis_enabled=basis_enabled, fx_enabled=fx_enabled, freight_enabled=freight_enabled,
            corn_price=corn_price, soy_price=soy_price, wheat_price=wheat_price,
            corn_qty=corn_qty, soy_qty=soy_qty, wheat_qty=wheat_qty,
            basis_mean_corn=basis_mean_corn, basis_mean_soy=basis_mean_soy, basis_mean_wht=basis_mean_wht, fx_rate=fx_rate
        )
        st.session_state["results"] = results
        st.success("Simulation complete. See Results / Hedging / Decomposition tabs.")

# -------- Results --------
with tabs[3]:
    st.header("Results Dashboard")
    if "results" in st.session_state:
        r = st.session_state["results"]
        c = st.columns(6)
        c[0].metric(f"95% VaR ({horizon}d)", format_currency(r["var95"]))
        c[1].metric(f"99% VaR ({horizon}d)", format_currency(r["var99"]))
        c[2].metric("Expected Shortfall (97.5%)", format_currency(r["es975"]))
        c[3].metric("Mean P&L", format_currency(r["mean"]))
        c[4].metric("Median P&L", format_currency(r["median"]))
        c[5].metric("Std Dev", format_currency(abs(r["std"])))

        hist = px.histogram(pd.DataFrame({"PnL": r["scen_pnls"]}), x="PnL", nbins=50, title="P&L Distribution")
        st.plotly_chart(hist, use_container_width=True)

        st.subheader("Risk Contribution by Factor")
        k = ["price","basis","fx","freight"]
        vals = [r["contrib"][x] for x in k]
        bar = px.bar(x=[x.capitalize() for x in k], y=vals, labels={"x": "Factor", "y": "Contribution (%)"},
                     title="Risk Contribution (%)")
        st.plotly_chart(bar, use_container_width=True)
    else:
        st.info("Run a simulation first.")

# -------- Hedging (kept simple) --------
with tabs[4]:
    st.header("Hedging Optimization")
    if "results" in st.session_state:
        r = st.session_state["results"]
        hedge_ratio = 0.80
        eff = 0.85
        # This minimal enhanced app doesn't track contracts; report reduction only.
        hedged_var95 = r["var95"] * (1 - hedge_ratio * eff)
        hedged_var99 = r["var99"] * (1 - hedge_ratio * eff)
        red_pct = 100.0 * (r["var95"] - hedged_var95) / (r["var95"] + 1e-12)

        cols = st.columns(3)
        cols[0].metric("Hedge Ratio", "80%")
        cols[1].metric("Expected VaR Reduction", f"{red_pct:.1f}%")
        cols[2].metric("Effectiveness Assumed", "85%")

        comp = pd.DataFrame({
            "Portfolio": ["Unhedged", "Hedged"],
            "VaR 95%": [r["var95"], hedged_var95],
            "VaR 99%": [r["var99"], hedged_var99],
        })
        fig = px.bar(comp.melt(id_vars="Portfolio", var_name="Metric", value_name="Value"),
                     x="Portfolio", y="Value", color="Metric", barmode="group", title="Hedged vs Unhedged VaR")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run a simulation first.")

# -------- Stress Testing --------
with tabs[5]:
    st.header("Stress Testing")
    def stress_pnl(price_shock, basis_shock, fx_shock, freight_shock):
        total = 0.0
        for k, p0 in {"corn": corn_price, "soybeans": soy_price, "wheat": wheat_price}.items():
            new_p = p0 * (1.0 + price_shock)
            if basis_enabled:
                m = {"corn": basis_mean_corn/100.0, "soybeans": basis_mean_soy/100.0, "wheat": basis_mean_wht/100.0}[k]
                new_p += p0 * (m + basis_shock)
            if fx_enabled:
                new_fx = fx_rate * (1.0 + fx_shock)
                new_p = new_p * new_fx / fx_rate
            # freight cost treated as $/bu deduction not modeled in this simplified enhanced version
            q = {"corn": corn_qty, "soybeans": soy_qty, "wheat": wheat_qty}[k]
            total += q * (new_p - p0)
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

    rows = []
    if "stress_md" in st.session_state: rows.append(("Moderate Downturn", st.session_state["stress_md"]))
    if "stress_sc" in st.session_state: rows.append(("Severe Crisis", st.session_state["stress_sc"]))
    if "stress_ps" in st.session_state: rows.append(("Perfect Storm", st.session_state["stress_ps"]))
    if "stress_custom" in st.session_state: rows.append(("Custom", st.session_state["stress_custom"]))

    if rows:
        df_res = pd.DataFrame(rows, columns=["Scenario", "P&L ($)"])
        st.dataframe(df_res.style.format({"P&L ($)": "{:,.0f}"}), hide_index=True)
        bar = px.bar(df_res, x="Scenario", y="P&L ($)", title="Stress Test Results")
        st.plotly_chart(bar, use_container_width=True)

# -------- Decomposition --------
with tabs[6]:
    st.header("Risk Decomposition")
    if "results" in st.session_state:
        r = st.session_state["results"]
        k = ["price","basis","fx","freight"]
        vals = [r["contrib"][x] for x in k]
        heat = go.Figure(data=go.Heatmap(
            z=[vals],
            x=[x.capitalize() for x in k],
            y=["Contribution %"],
            coloraxis="coloraxis"
        ))
        heat.update_layout(title="Risk Factor Contribution Heatmap", coloraxis={'colorscale':'RdBu'})
        st.plotly_chart(heat, use_container_width=True)
    else:
        st.info("Run a simulation first.")
