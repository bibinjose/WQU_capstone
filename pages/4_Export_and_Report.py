import streamlit as st
import pandas as pd

st.set_page_config(page_title="Export & Report", layout="wide")
st.title("Export & Report")

if "results" not in st.session_state:
    st.info("Run a simulation on the main page first.")
else:
    r = st.session_state["results"]
    df = pd.DataFrame({"scenario_pnl": r["scen_pnls"]})
    st.dataframe(df.head(20), use_container_width=True)

    csv = df.to_csv(index=False).encode()
    st.download_button("Download Scenario P&L (CSV)", csv, file_name="scenario_pnl.csv", mime="text/csv")

    report = f"""Commodity Risk Engine Report

Horizon: {r['horizon']} days
VaR 95%: ${r['var95']:,.0f}
VaR 99%: ${r['var99']:,.0f}
ES 97.5%: ${r['es975']:,.0f}
Mean: ${r['mean']:,.0f} | Median: ${r['median']:,.0f} | Std: ${r['std']:,.0f}

Contributions (% of proxy):
  Price: {r['contrib']['price']:.1f}%
  Basis: {r['contrib']['basis']:.1f}%
  FX: {r['contrib']['fx']:.1f}%
  Freight: {r['contrib']['freight']:.1f}%
"""
    st.download_button("Download Text Report", report.encode(), file_name="risk_report.txt", mime="text/plain")
    st.success("Exports ready.")
