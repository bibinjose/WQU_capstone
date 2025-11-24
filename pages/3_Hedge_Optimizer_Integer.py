import streamlit as st
import numpy as np

st.set_page_config(page_title="Hedge Optimizer (Integer)", layout="wide")
st.title("Hedge Optimizer (Integer contracts)")
st.caption("Minimize ES(97.5%) across integer futures contracts using your current parameters.")

if "results" not in st.session_state:
    st.info("Run a simulation on the main page first so we can reuse parameters.")
else:
    r = st.session_state["results"]
    params = st.session_state.get("params", {})
    prices = r["prices"]; qtys = r["qtys"]; cs = r["contract_sizes"]
    horizon = r["horizon"]

    price_vol = float(params.get("price_vol", 18.0))/100.0/np.sqrt(252.0) * np.sqrt(horizon)
    rng = st.number_input("Scenarios", 5000, 100000, 20000, 5000)
    step = st.select_slider("Grid step", options=[1,2,5,10,20], value=10)

    maxC = int(qtys["corn"]//cs["corn"]); maxS = int(qtys["soybeans"]//cs["soybeans"]); maxW = int(qtys["wheat"]//cs["wheat"])
    st.caption(f"Contract search bounds: Corn 0..{maxC}, Soy 0..{maxS}, Wheat 0..{maxW}")

    best=None; best_es=1e99
    if st.button("Run optimizer"):
        for C in range(0, maxC+1, step):
            for S in range(0, maxS+1, step):
                for W in range(0, maxW+1, step):
                    dF_c = prices["corn"]*np.random.normal(scale=price_vol, size=rng)
                    dF_s = prices["soybeans"]*np.random.normal(scale=price_vol, size=rng)
                    dF_w = prices["wheat"]*np.random.normal(scale=price_vol, size=rng)
                    phys = dF_c*qtys["corn"] + dF_s*qtys["soybeans"] + dF_w*qtys["wheat"]
                    hedge = -(dF_c*cs["corn"]*C + dF_s*cs["soybeans"]*S + dF_w*cs["wheat"]*W)
                    pnl = phys + hedge
                    q = max(1, int(0.025*len(pnl)))
                    es = -np.sort(pnl)[:q].mean()
                    if es < best_es:
                        best_es = float(es); best=(C,S,W)
        if best:
            st.success(f"Optimal ≈ Corn {best[0]}, Soy {best[1]}, Wheat {best[2]}  →  ES97.5 ≈ ${best_es:,.0f}")
            st.session_state["optimizer_best"] = dict(corn=best[0], soybeans=best[1], wheat=best[2], es975=best_es)
