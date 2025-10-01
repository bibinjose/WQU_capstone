# WQU_capstone
Value-at-Risk (VaR) and Expected Shortfall (ES) in Commodities

Price reference: CBOT/CME futures (continuous, rolled series).

VaR, ES, risk attribution, stress tests, liquidity-adjusted VaR.


Process and problem statment
1. converts production + inventory into stochastic revenue/margin using futures (with an option to add basis/FX/freight);
2. computes VaR, ES, and CFaR on user-defined horizons;
4. sizes regime-aware hedges to meet a CFaR target;
5. decomposes risk by driver (price, basis, FX, freight);
6. runs stress & liquidity tests;
7. reports unhedged vs hedged outcomes
