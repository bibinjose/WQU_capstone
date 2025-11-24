import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Historical Calibration", layout="wide")
st.title("Historical Calibration")
st.caption("Pull futures history via Yahoo Finance to suggest vols, correlations, and Student-t df.")
st.caption(" Student Group 11203-Bibin Jose,Nicolas Vidal,Freddy Kuriakose")


with st.sidebar:
    years = st.slider("Lookback (years)", 3, 12, 5)
    use_ewma = st.checkbox("Use EWMA scaling for vol", value=False)
    lam = st.slider("EWMA λ", 0.80, 0.99, 0.94, 0.01)

def ewma_vol(series, lam=0.94):
    v = 0.0
    for x in series.dropna().values:
        v = lam*v + (1-lam)*(x**2)
    return float(np.sqrt(v))

try:
    import yfinance as yf
    st.info("Downloading ZC=F (corn), ZS=F (soy), ZW=F (wheat)…")
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
    ret = dfp.pct_change().dropna()

    vol_ann = ret.std()*np.sqrt(252)*100
    corr = ret.corr()

    price_vol = float(vol_ann.mean())
    if use_ewma:
        ew = np.mean([ewma_vol(ret[c], lam)*np.sqrt(252)*100 for c in ["corn","soybeans","wheat"]])
        price_vol = float(ew)

    kurt = float(ret.kurt().mean())
    df_suggest = max(3, int(6.0/kurt + 4.0)) if kurt>0.1 else 8

    c1,c2,c3 = st.columns(3)
    c1.metric("Suggested Price Vol (%)", f"{price_vol:.1f}")
    c2.metric("Suggested t-df", f"{df_suggest}")
    c3.metric("Avg Kurtosis", f"{kurt:.2f}")

    st.subheader("Correlation (returns)")
    st.dataframe(corr.style.format("{:.2f}"), use_container_width=True)

    if st.button("Apply to main app"):
        st.session_state.setdefault("params",{})
        st.session_state["params"]["price_vol"] = price_vol
        st.session_state["params"]["corr_cs"] = float(corr.loc["corn","soybeans"])
        st.session_state["params"]["corr_cw"] = float(corr.loc["corn","wheat"])
        st.session_state["params"]["corr_sw"] = float(corr.loc["soybeans","wheat"])
        st.success("Applied. Go to the main app → Risk Factors.")
except Exception as e:
    st.warning(f"Historical download unavailable ({e}). Install yfinance and ensure internet.")


