import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

st.set_page_config(page_title="Backtesting & Kupiec", layout="wide")
st.title("Backtesting & Kupiec Test")
st.caption("Rolling historical VaR on futures P&L proxy, plus Kupiec proportion-of-failures test.")
st.caption(" Student Group 11203-Bibin Jose,Nicolas Vidal,Freddy Kuriakose")

with st.sidebar:
    years = st.slider("Lookback (years)", 3, 12, 5)
    win = st.number_input("Rolling window (days)", 100, 500, 250, 10)
    alpha = st.select_slider("VaR Tail", [0.90,0.95,0.975,0.99], value=0.95)

def kupiec_pof(fails, N, alpha):
    if N<=0: return float("nan"), float("nan")
    x = fails; p = 1-alpha
    if x==0 or x==N: x = max(1, min(N-1, x))
    term1 = (1-p)**(N-x) * (p**x)
    pi = x/N
    term2 = (1-pi)**(N-x) * (pi**x)
    LR = -2.0 * (np.log(term1) - np.log(term2))
    from math import erf, sqrt
    return float(LR), float(1 - erf(np.sqrt(LR/2.0)))

try:
    import yfinance as yf
    st.info("Downloading ZC=F, ZS=F, ZW=F…")
    px = {}
    for tkr,name in [("ZC=F","corn"),("ZS=F","soybeans"),("ZW=F","wheat")]:
        df = yf.download(tkr, period=f"{years}y", auto_adjust=True, progress=False)

        # Get the Close as a Series robustly
        s = df["Close"]
        if isinstance(s, pd.DataFrame):      # sometimes still DataFrame with a single col
            s = s.iloc[:, 0]
        s = s.astype("float64")              # ensure numeric
        s.name = name                        # set Series name (don't use .rename(name))

        px[name] = s
    dfp = pd.concat(px, axis=1).dropna()
    dpx = dfp.diff().dropna()

    # Pull your latest simulated quantities/contracts if available
    q = st.session_state.get("results",{}).get("qtys", {"corn":0,"soybeans":0,"wheat":0})
    cs = st.session_state.get("results",{}).get("contract_sizes", {"corn":5000,"soybeans":5000,"wheat":5000})
    h = {"corn":0,"soybeans":0,"wheat":0}  # set here if you want to include hedge backtests

    pnl_hist = (dpx["corn"]*q["corn"] + dpx["soybeans"]*q["soybeans"] + dpx["wheat"]*q["wheat"])
    hedge_pnl = -(dpx["corn"]*cs["corn"]*h["corn"] + dpx["soybeans"]*cs["soybeans"]*h["soybeans"] + dpx["wheat"]*cs["wheat"]*h["wheat"])
    pnl = (pnl_hist + hedge_pnl).dropna()

    losses = -pnl
    VaR = losses.rolling(win).quantile(alpha).shift(1)
    test_slice = losses.iloc[win:]
    VaR_slice = VaR.iloc[win:]
    fails = (test_slice > VaR_slice).sum()
    N = len(test_slice.dropna())
    LR, pval = kupiec_pof(int(fails), int(N), alpha)

    #st.line_chart(pd.DataFrame({"Loss": losses, f"VaR@{alpha:.3f}": VaR}), use_container_width=True)


    # Create DataFrame for plotting
    plot_df = pd.DataFrame({"Loss": losses, f"VaR@{alpha:.3f}": VaR})

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(plot_df.index, plot_df["Loss"], label="Loss")
    ax.plot(plot_df.index, plot_df[f"VaR@{alpha:.3f}"], label=f"VaR@{alpha:.3f}", alpha=0.5)

    # 🗓️ Format X-axis to show Year–Month ticks
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%b"))
    fig.autofmt_xdate()

    ax.set_ylabel("P&L ($)")
    ax.set_title("Historical Losses vs VaR Threshold")
    ax.legend()

    st.pyplot(fig, width="stretch")


    if N>0:
        st.success(f"Kupiec POF: failures={int(fails)} over N={N}, LR={LR:.2f}, p≈{pval:.3f}")
    else:
        st.warning("Not enough data for the selected window.")
except Exception as e:
    st.warning(f"Backtest unavailable ({e}). Install yfinance and ensure internet.")

