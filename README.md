# WQU_capstone
Value-at-Risk (VaR) and Expected Shortfall (ES) in Commodities

Price reference: CBOT/CME futures (continuous, rolled series).

VaR, ES, risk attribution, stress tests, liquidity-adjusted VaR.


Process and problem statement
1. converts production + inventory into stochastic revenue/margin using futures (with an option to add basis/FX/freight);
2. computes VaR, ES, and CFaR on user-defined horizons;
4. sizes regime-aware hedges to meet a CFaR target;
5. decomposes risk by driver (price, basis, FX, freight);
6. runs stress & liquidity tests;
7. reports unhedged vs hedged outcomes


## Production+Inventory to stochastic model
We can map the production using the CBOT price, adjusting for the local basis, the FX price, export flows, and storage costs.
We use a non-stochastic model since the Commodities follow a non-Gaussian distribution. For that reason, we use Student-t and GARCH volatility or a filter of historical residual. We also have a dependency on time-varying values and the correlation of commodities, FX, and basis.

## VAR, ES, and CFAR computation
VAR is used as a standard to measure the tail quantile. 
ES is the tail mean
We take the VAR/ES portfolio and transition to CFaR to plan for liquidity and available cash.


## Establish the Regime hedge size to meet the CFaR target
We establish the structural situation. WE establish the risk and exposure of carrying.
We establish the financial conditions and monetary policy.
We establish the trend and volatility regime.

After deciding on a CFaR target (80% Reduction vs unhedged case) to determine the minimum contracts so that CFaR is less than the target under the established regime.

## Drivers decompose Risk
We decompose the revenue CBOT price + basis + FX + freight + carry. We then show in hedge vs Unhedge choosing drivers that establish hedge nature.

## Stress testing and Liquidity adjust
We run different stress test: Historical, Multisigma Shortfall, and other fundamentals, Policy shock, and Liquidity-adjust VaR.

## Unhedge vs Hedge Outcomes
We show distributions from P&L, Var/ES/CFaR, stress tables and risk drivers in hedge and unhedge porfolios




