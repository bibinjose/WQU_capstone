// Commodity Risk Engine - Monte Carlo Simulation
class CommodityRiskEngine {
  constructor() {
    this.portfolio = {
      corn: { qty: 1000000, price: 4.50, symbol: 'ZC', contractSize: 5000 },
      soybeans: { qty: 600000, price: 11.50, symbol: 'ZS', contractSize: 5000 },
      wheat: { qty: 800000, price: 5.50, symbol: 'ZW', contractSize: 5000 }
    };
    
    this.riskFactors = {
      price: { enabled: true, vol: 0.18 },
      basis: { enabled: true, vol: 0.05, corr: 0.3, mean: { corn: -0.05, soybeans: -0.08, wheat: -0.06 } },
      fx: { enabled: true, pair: 'USD/BRL', rate: 5.20, vol: 0.15, corr: -0.20 },
      freight: { enabled: true, cost: 0.50, vol: 0.35, corr: 0.25 }
    };
    
    this.correlations = {
      cornSoy: 0.47,
      cornWheat: 0.37,
      soyWheat: 0.41
    };
    
    this.simulationResults = null;
    this.charts = {};
    
    this.initializeUI();
  }
  
  initializeUI() {
    // Tab navigation
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => this.switchTab(btn.dataset.tab));
    });
    
    // Risk factor toggles
    ['price', 'basis', 'fx', 'freight'].forEach(factor => {
      const toggle = document.getElementById(`${factor}-enabled`);
      const config = document.getElementById(`${factor}-config`);
      toggle.addEventListener('change', () => {
        this.riskFactors[factor].enabled = toggle.checked;
        config.classList.toggle('disabled', !toggle.checked);
        this.updateRiskStatus();
      });
    });
    
    // FX pair selector
    document.getElementById('fx-pair').addEventListener('change', (e) => {
      const rates = {
        'USD/BRL': { rate: 5.20, vol: 15, corr: -0.20 },
        'USD/CAD': { rate: 1.35, vol: 8, corr: -0.15 },
        'USD/INR': { rate: 83.50, vol: 6, corr: 0.05 },
        'EUR/USD': { rate: 1.08, vol: 10, corr: -0.10 }
      };
      const pair = rates[e.target.value];
      document.getElementById('fx-rate').value = pair.rate;
      document.getElementById('fx-vol').value = pair.vol;
      document.getElementById('fx-corr').value = pair.corr;
    });
    
    // Distribution selector
    document.getElementById('distribution').addEventListener('change', (e) => {
      document.getElementById('df-group').style.display = 
        e.target.value === 'student_t' ? 'block' : 'none';
    });
    
    // Custom stress sliders
    ['price', 'basis', 'fx', 'freight'].forEach(factor => {
      const slider = document.getElementById(`custom-${factor}`);
      const display = document.getElementById(`custom-${factor}-val`);
      slider.addEventListener('input', () => {
        display.textContent = slider.value + '%';
      });
    });
    
    // Initialize portfolio chart
    this.updatePortfolio();
    this.updateRiskStatus();
  }
  
  switchTab(tabName) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
    
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
    document.getElementById(`${tabName}-tab`).classList.add('active');
  }
  
  updatePortfolio() {
    // Get values from inputs
    this.portfolio.corn.qty = parseFloat(document.getElementById('corn-qty').value);
    this.portfolio.corn.price = parseFloat(document.getElementById('corn-price').value);
    this.portfolio.soybeans.qty = parseFloat(document.getElementById('soy-qty').value);
    this.portfolio.soybeans.price = parseFloat(document.getElementById('soy-price').value);
    this.portfolio.wheat.qty = parseFloat(document.getElementById('wheat-qty').value);
    this.portfolio.wheat.price = parseFloat(document.getElementById('wheat-price').value);
    
    // Calculate values
    const cornValue = this.portfolio.corn.qty * this.portfolio.corn.price;
    const soyValue = this.portfolio.soybeans.qty * this.portfolio.soybeans.price;
    const wheatValue = this.portfolio.wheat.qty * this.portfolio.wheat.price;
    const totalValue = cornValue + soyValue + wheatValue;
    
    // Update display
    document.getElementById('total-value').textContent = this.formatCurrency(totalValue);
    document.getElementById('corn-value').textContent = this.formatCurrency(cornValue, true);
    document.getElementById('soy-value').textContent = this.formatCurrency(soyValue, true);
    document.getElementById('wheat-value').textContent = this.formatCurrency(wheatValue, true);
    
    document.getElementById('corn-pct').textContent = ((cornValue / totalValue) * 100).toFixed(1) + '%';
    document.getElementById('soy-pct').textContent = ((soyValue / totalValue) * 100).toFixed(1) + '%';
    document.getElementById('wheat-pct').textContent = ((wheatValue / totalValue) * 100).toFixed(1) + '%';
    
    // Update chart
    this.renderPortfolioChart();
  }
  
  renderPortfolioChart() {
    const ctx = document.getElementById('portfolio-chart');
    if (this.charts.portfolio) this.charts.portfolio.destroy();
    
    const cornValue = this.portfolio.corn.qty * this.portfolio.corn.price;
    const soyValue = this.portfolio.soybeans.qty * this.portfolio.soybeans.price;
    const wheatValue = this.portfolio.wheat.qty * this.portfolio.wheat.price;
    
    this.charts.portfolio = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: ['Corn', 'Soybeans', 'Wheat'],
        datasets: [{
          data: [cornValue, soyValue, wheatValue],
          backgroundColor: ['#FFB700', '#6B8E23', '#D2691E'],
          borderWidth: 2,
          borderColor: getComputedStyle(document.documentElement).getPropertyValue('--color-surface')
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'bottom' },
          title: { display: true, text: 'Portfolio Composition' }
        }
      }
    });
  }
  
  updateRiskStatus() {
    const statusDiv = document.getElementById('risk-status');
    const factors = [
      {
        name: 'Price Risk',
        enabled: this.riskFactors.price.enabled,
        desc: `${document.getElementById('price-vol').value}% annualized volatility`
      },
      {
        name: 'Basis Risk',
        enabled: this.riskFactors.basis.enabled,
        desc: `${document.getElementById('basis-vol').value}% volatility, ${document.getElementById('basis-corr').value} correlation`
      },
      {
        name: 'FX Risk',
        enabled: this.riskFactors.fx.enabled,
        desc: `${document.getElementById('fx-pair').value} at ${document.getElementById('fx-rate').value}, ${document.getElementById('fx-vol').value}% volatility`
      },
      {
        name: 'Freight Risk',
        enabled: this.riskFactors.freight.enabled,
        desc: `$${document.getElementById('freight-cost').value}/bu, ${document.getElementById('freight-vol').value}% volatility`
      }
    ];
    
    statusDiv.innerHTML = factors.map(f => `
      <div class="status-item">
        <span class="status ${f.enabled ? 'status--success' : 'status--error'}">
          ${f.enabled ? 'Enabled' : 'Disabled'}
        </span>
        <strong>${f.name}:</strong> ${f.desc}
      </div>
    `).join('');
  }
  
  async runSimulation() {
    const btn = document.getElementById('run-btn');
    const btnText = document.getElementById('btn-text');
    const btnLoading = document.getElementById('btn-loading');
    
    btn.disabled = true;
    btnText.classList.add('hidden');
    btnLoading.classList.remove('hidden');
    
    // Small delay to show loading state
    await new Promise(resolve => setTimeout(resolve, 100));
    
    try {
      // Get simulation parameters
      const numScenarios = parseInt(document.getElementById('num-scenarios').value);
      const horizon = parseInt(document.getElementById('time-horizon').value);
      const distribution = document.getElementById('distribution').value;
      const df = parseInt(document.getElementById('degrees-freedom').value);
      
      // Get risk factor parameters
      const priceVol = parseFloat(document.getElementById('price-vol').value) / 100;
      const basisVol = parseFloat(document.getElementById('basis-vol').value) / 100;
      const basisCorr = parseFloat(document.getElementById('basis-corr').value);
      const fxVol = parseFloat(document.getElementById('fx-vol').value) / 100;
      const fxCorr = parseFloat(document.getElementById('fx-corr').value);
      const fxRate = parseFloat(document.getElementById('fx-rate').value);
      const freightCost = parseFloat(document.getElementById('freight-cost').value);
      const freightVol = parseFloat(document.getElementById('freight-vol').value) / 100;
      const freightCorr = parseFloat(document.getElementById('freight-corr').value);
      
      // Get commodity correlations
      const corrCS = parseFloat(document.getElementById('corr-corn-soy').value);
      const corrCW = parseFloat(document.getElementById('corr-corn-wheat').value);
      const corrSW = parseFloat(document.getElementById('corr-soy-wheat').value);
      
      // Get basis means
      const basisMean = {
        corn: parseFloat(document.getElementById('basis-corn').value) / 100,
        soybeans: parseFloat(document.getElementById('basis-soy').value) / 100,
        wheat: parseFloat(document.getElementById('basis-wheat').value) / 100
      };
      
      // Calculate daily volatilities
      const dailyPriceVol = priceVol / Math.sqrt(252);
      const dailyBasisVol = basisVol / Math.sqrt(252);
      const dailyFxVol = fxVol / Math.sqrt(252);
      const dailyFreightVol = freightVol / Math.sqrt(252);
      
      // Scale by time horizon
      const scaledPriceVol = dailyPriceVol * Math.sqrt(horizon);
      const scaledBasisVol = dailyBasisVol * Math.sqrt(horizon);
      const scaledFxVol = dailyFxVol * Math.sqrt(horizon);
      const scaledFreightVol = dailyFreightVol * Math.sqrt(horizon);
      
      // Build correlation matrix for all factors
      // Order: Corn Price, Soy Price, Wheat Price, Corn Basis, Soy Basis, Wheat Basis, FX, Freight
      const corrMatrix = [
        [1, corrCS, corrCW, basisCorr, 0, 0, fxCorr, freightCorr],
        [corrCS, 1, corrSW, 0, basisCorr, 0, fxCorr, freightCorr],
        [corrCW, corrSW, 1, 0, 0, basisCorr, fxCorr, freightCorr],
        [basisCorr, 0, 0, 1, 0, 0, 0, 0],
        [0, basisCorr, 0, 0, 1, 0, 0, 0],
        [0, 0, basisCorr, 0, 0, 1, 0, 0],
        [fxCorr, fxCorr, fxCorr, 0, 0, 0, 1, 0],
        [freightCorr, freightCorr, freightCorr, 0, 0, 0, 0, 1]
      ];
      
      // Cholesky decomposition
      const cholesky = this.choleskyDecomposition(corrMatrix);
      
      // Generate scenarios
      const scenarios = [];
      for (let i = 0; i < numScenarios; i++) {
        // Generate independent random variables
        let randoms = [];
        for (let j = 0; j < 8; j++) {
          randoms.push(this.randomNormal());
        }
        
        // Apply Student-t if selected
        if (distribution === 'student_t') {
          const chiSq = this.randomChiSquare(df);
          const scale = Math.sqrt(df / chiSq);
          randoms = randoms.map(r => r * scale);
        }
        
        // Apply Cholesky to get correlated returns
        const correlated = this.multiplyMatrixVector(cholesky, randoms);
        
        // Extract returns for each factor
        const priceReturns = {
          corn: correlated[0] * scaledPriceVol,
          soybeans: correlated[1] * scaledPriceVol,
          wheat: correlated[2] * scaledPriceVol
        };
        
        const basisReturns = {
          corn: correlated[3] * scaledBasisVol,
          soybeans: correlated[4] * scaledBasisVol,
          wheat: correlated[5] * scaledBasisVol
        };
        
        const fxReturn = correlated[6] * scaledFxVol;
        const freightReturn = correlated[7] * scaledFreightVol;
        
        // Calculate scenario P&L
        let totalPnL = 0;
        const commodities = ['corn', 'soybeans', 'wheat'];
        
        for (const commodity of commodities) {
          const position = this.portfolio[commodity];
          const priceRet = priceReturns[commodity];
          const basisRet = basisReturns[commodity];
          
          // New price after return
          let newPrice = position.price * (1 + priceRet);
          
          // Add basis if enabled
          if (this.riskFactors.basis.enabled) {
            const basisChange = position.price * (basisMean[commodity] + basisRet);
            newPrice += basisChange;
          }
          
          // Apply FX if enabled
          if (this.riskFactors.fx.enabled) {
            const newFxRate = fxRate * (1 + fxReturn);
            newPrice = newPrice * newFxRate / fxRate;
          }
          
          // Deduct freight if enabled
          if (this.riskFactors.freight.enabled) {
            const newFreight = freightCost * (1 + freightReturn);
            newPrice -= newFreight;
          }
          
          // Calculate P&L for this commodity
          const pnl = position.qty * (newPrice - position.price);
          totalPnL += pnl;
        }
        
        scenarios.push({
          pnl: totalPnL,
          priceReturns,
          basisReturns,
          fxReturn,
          freightReturn
        });
      }
      
      // Sort scenarios by P&L
      scenarios.sort((a, b) => a.pnl - b.pnl);
      
      // Calculate VaR and ES
      const conf95 = document.getElementById('conf-95').checked;
      const conf99 = document.getElementById('conf-99').checked;
      
      const var95 = conf95 ? -scenarios[Math.floor(numScenarios * 0.05)].pnl : null;
      const var99 = conf99 ? -scenarios[Math.floor(numScenarios * 0.01)].pnl : null;
      
      // ES at 97.5% (average of worst 2.5%)
      const esIndex = Math.floor(numScenarios * 0.025);
      const esPnLs = scenarios.slice(0, esIndex).map(s => s.pnl);
      const es975 = -esPnLs.reduce((a, b) => a + b, 0) / esPnLs.length;
      
      // Calculate statistics
      const pnls = scenarios.map(s => s.pnl);
      const mean = pnls.reduce((a, b) => a + b, 0) / pnls.length;
      const median = pnls[Math.floor(numScenarios / 2)];
      const variance = pnls.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / pnls.length;
      const stdDev = Math.sqrt(variance);
      
      // Calculate risk contributions
      const contributions = this.calculateRiskContributions(scenarios);
      
      // Store results
      this.simulationResults = {
        scenarios,
        var95,
        var99,
        es975,
        mean,
        median,
        stdDev,
        contributions,
        numScenarios,
        horizon
      };
      
      // Display results
      this.displayResults();
      this.displayHedging();
      this.displayDecomposition();
      
      // Switch to results tab
      this.switchTab('results');
      
    } finally {
      btn.disabled = false;
      btnText.classList.remove('hidden');
      btnLoading.classList.add('hidden');
    }
  }
  
  calculateRiskContributions(scenarios) {
    // Run simulations with each factor disabled to measure contribution
    const baseVar = -scenarios[Math.floor(scenarios.length * 0.05)].pnl;
    
    // Approximate contributions based on variance decomposition
    const priceVariance = scenarios.reduce((sum, s) => {
      const priceComponent = Object.values(s.priceReturns).reduce((a, b) => a + Math.abs(b), 0);
      return sum + Math.pow(priceComponent, 2);
    }, 0);
    
    const basisVariance = scenarios.reduce((sum, s) => {
      const basisComponent = Object.values(s.basisReturns).reduce((a, b) => a + Math.abs(b), 0);
      return sum + Math.pow(basisComponent, 2);
    }, 0);
    
    const fxVariance = scenarios.reduce((sum, s) => {
      return sum + Math.pow(s.fxReturn, 2);
    }, 0);
    
    const freightVariance = scenarios.reduce((sum, s) => {
      return sum + Math.pow(s.freightReturn, 2);
    }, 0);
    
    const total = priceVariance + basisVariance + fxVariance + freightVariance;
    
    return {
      price: (priceVariance / total) * 100,
      basis: (basisVariance / total) * 100,
      fx: (fxVariance / total) * 100,
      freight: (freightVariance / total) * 100
    };
  }
  
  displayResults() {
    const r = this.simulationResults;
    const resultsDiv = document.getElementById('results-content');
    
    let html = '<div class="results-grid">';
    
    if (r.var95) {
      html += `
        <div class="result-card">
          <h4>95% VaR (${r.horizon} days)</h4>
          <div class="value negative">${this.formatCurrency(r.var95)}</div>
          <div class="subtitle">Maximum expected loss at 95% confidence</div>
        </div>
      `;
    }
    
    if (r.var99) {
      html += `
        <div class="result-card">
          <h4>99% VaR (${r.horizon} days)</h4>
          <div class="value negative">${this.formatCurrency(r.var99)}</div>
          <div class="subtitle">Maximum expected loss at 99% confidence</div>
        </div>
      `;
    }
    
    html += `
      <div class="result-card">
        <h4>Expected Shortfall (97.5%)</h4>
        <div class="value negative">${this.formatCurrency(r.es975)}</div>
        <div class="subtitle">Average loss in worst 2.5% scenarios</div>
      </div>
      <div class="result-card">
        <h4>Mean P&amp;L</h4>
        <div class="value ${r.mean >= 0 ? 'positive' : 'negative'}">${this.formatCurrency(r.mean)}</div>
        <div class="subtitle">Expected portfolio change</div>
      </div>
      <div class="result-card">
        <h4>Median P&amp;L</h4>
        <div class="value ${r.median >= 0 ? 'positive' : 'negative'}">${this.formatCurrency(r.median)}</div>
        <div class="subtitle">50th percentile outcome</div>
      </div>
      <div class="result-card">
        <h4>Standard Deviation</h4>
        <div class="value">${this.formatCurrency(Math.abs(r.stdDev))}</div>
        <div class="subtitle">Portfolio volatility</div>
      </div>
    </div>
    
    <div class="card">
      <h3>Risk Contribution by Factor</h3>
      <div class="risk-contribution">
        <div class="contribution-item">
          <div class="contribution-label">Price Risk</div>
          <div class="contribution-bar">
            <div class="contribution-fill" style="width: ${r.contributions.price}%"></div>
          </div>
          <div class="contribution-value">${r.contributions.price.toFixed(1)}%</div>
        </div>
        <div class="contribution-item">
          <div class="contribution-label">Basis Risk</div>
          <div class="contribution-bar">
            <div class="contribution-fill" style="width: ${r.contributions.basis}%"></div>
          </div>
          <div class="contribution-value">${r.contributions.basis.toFixed(1)}%</div>
        </div>
        <div class="contribution-item">
          <div class="contribution-label">FX Risk</div>
          <div class="contribution-bar">
            <div class="contribution-fill" style="width: ${r.contributions.fx}%"></div>
          </div>
          <div class="contribution-value">${r.contributions.fx.toFixed(1)}%</div>
        </div>
        <div class="contribution-item">
          <div class="contribution-label">Freight Risk</div>
          <div class="contribution-bar">
            <div class="contribution-fill" style="width: ${r.contributions.freight}%"></div>
          </div>
          <div class="contribution-value">${r.contributions.freight.toFixed(1)}%</div>
        </div>
      </div>
    </div>
    
    <div class="card chart-card" style="min-height: 400px;">
      <canvas id="pnl-histogram"></canvas>
    </div>
    `;
    
    resultsDiv.innerHTML = html;
    
    // Render histogram
    setTimeout(() => this.renderPnLHistogram(), 100);
  }
  
  renderPnLHistogram() {
    const ctx = document.getElementById('pnl-histogram');
    if (!ctx) return;
    
    const r = this.simulationResults;
    const pnls = r.scenarios.map(s => s.pnl);
    
    // Create histogram bins
    const numBins = 50;
    const min = Math.min(...pnls);
    const max = Math.max(...pnls);
    const binWidth = (max - min) / numBins;
    
    const bins = new Array(numBins).fill(0);
    const binLabels = [];
    
    for (let i = 0; i < numBins; i++) {
      const binStart = min + i * binWidth;
      binLabels.push(this.formatCurrency(binStart, true));
      
      for (const pnl of pnls) {
        if (pnl >= binStart && pnl < binStart + binWidth) {
          bins[i]++;
        }
      }
    }
    
    if (this.charts.histogram) this.charts.histogram.destroy();
    
    this.charts.histogram = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: binLabels,
        datasets: [{
          label: 'Frequency',
          data: bins,
          backgroundColor: '#1FB8CD',
          borderWidth: 0
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: { display: true, text: 'P&L Distribution' },
          legend: { display: false },
          tooltip: {
            callbacks: {
              title: (items) => `P&L: ${items[0].label}`,
              label: (item) => `Scenarios: ${item.parsed.y}`
            }
          }
        },
        scales: {
          x: { 
            title: { display: true, text: 'P&L' },
            ticks: { maxTicksLimit: 10 }
          },
          y: { title: { display: true, text: 'Frequency' } }
        }
      }
    });
  }
  
  displayHedging() {
    if (!this.simulationResults) return;
    
    const r = this.simulationResults;
    const hedgeDiv = document.getElementById('hedging-content');
    
    // Calculate optimal hedge ratios based on risk contribution
    const totalValue = this.portfolio.corn.qty * this.portfolio.corn.price +
                       this.portfolio.soybeans.qty * this.portfolio.soybeans.price +
                       this.portfolio.wheat.qty * this.portfolio.wheat.price;
    
    // Simple hedge ratio: proportion of value to hedge
    const hedgeRatio = 0.8; // 80% hedge
    
    const cornContracts = Math.round((this.portfolio.corn.qty * hedgeRatio) / this.portfolio.corn.contractSize);
    const soyContracts = Math.round((this.portfolio.soybeans.qty * hedgeRatio) / this.portfolio.soybeans.contractSize);
    const wheatContracts = Math.round((this.portfolio.wheat.qty * hedgeRatio) / this.portfolio.wheat.contractSize);
    
    // Estimate hedged VaR (simplified: reduce by hedge ratio)
    const hedgedVar95 = r.var95 * (1 - hedgeRatio * 0.85); // 85% effectiveness
    const hedgedVar99 = r.var99 * (1 - hedgeRatio * 0.85);
    const reduction = ((r.var95 - hedgedVar95) / r.var95 * 100).toFixed(1);
    
    const html = `
      <div class="card">
        <h3>Optimal Hedge Strategy</h3>
        <p class="help-text">Based on risk contribution analysis, the following hedge positions are recommended:</p>
        
        <div class="results-grid">
          <div class="result-card">
            <h4>Corn (${this.portfolio.corn.symbol})</h4>
            <div class="value">${cornContracts}</div>
            <div class="subtitle">Contracts to short (${this.portfolio.corn.contractSize} bu each)</div>
          </div>
          <div class="result-card">
            <h4>Soybeans (${this.portfolio.soybeans.symbol})</h4>
            <div class="value">${soyContracts}</div>
            <div class="subtitle">Contracts to short (${this.portfolio.soybeans.contractSize} bu each)</div>
          </div>
          <div class="result-card">
            <h4>Wheat (${this.portfolio.wheat.symbol})</h4>
            <div class="value">${wheatContracts}</div>
            <div class="subtitle">Contracts to short (${this.portfolio.wheat.contractSize} bu each)</div>
          </div>
          <div class="result-card">
            <h4>Hedge Effectiveness</h4>
            <div class="value positive">${reduction}%</div>
            <div class="subtitle">Expected VaR reduction</div>
          </div>
        </div>
      </div>
      
      <div class="hedge-comparison">
        <div class="hedge-column">
          <h4>Unhedged Portfolio</h4>
          <div class="result-card">
            <h4>95% VaR</h4>
            <div class="value negative">${this.formatCurrency(r.var95)}</div>
          </div>
          <div class="result-card">
            <h4>99% VaR</h4>
            <div class="value negative">${this.formatCurrency(r.var99)}</div>
          </div>
        </div>
        <div class="hedge-column">
          <h4>Hedged Portfolio</h4>
          <div class="result-card">
            <h4>95% VaR</h4>
            <div class="value negative">${this.formatCurrency(hedgedVar95)}</div>
            <div class="subtitle positive">↓ ${((r.var95 - hedgedVar95) / 1000000).toFixed(2)}M reduction</div>
          </div>
          <div class="result-card">
            <h4>99% VaR</h4>
            <div class="value negative">${this.formatCurrency(hedgedVar99)}</div>
            <div class="subtitle positive">↓ ${((r.var99 - hedgedVar99) / 1000000).toFixed(2)}M reduction</div>
          </div>
        </div>
      </div>
    `;
    
    hedgeDiv.innerHTML = html;
  }
  
  displayDecomposition() {
    if (!this.simulationResults) return;
    
    const r = this.simulationResults;
    const decompDiv = document.getElementById('decomposition-content');
    
    // Calculate marginal VaR by commodity
    const totalValue = this.portfolio.corn.qty * this.portfolio.corn.price +
                       this.portfolio.soybeans.qty * this.portfolio.soybeans.price +
                       this.portfolio.wheat.qty * this.portfolio.wheat.price;
    
    const cornWeight = (this.portfolio.corn.qty * this.portfolio.corn.price) / totalValue;
    const soyWeight = (this.portfolio.soybeans.qty * this.portfolio.soybeans.price) / totalValue;
    const wheatWeight = (this.portfolio.wheat.qty * this.portfolio.wheat.price) / totalValue;
    
    // Simplified marginal VaR (weighted by exposure and volatility)
    const priceVol = parseFloat(document.getElementById('price-vol').value) / 100;
    const cornMarginal = cornWeight * priceVol * r.var95;
    const soyMarginal = soyWeight * priceVol * r.var95;
    const wheatMarginal = wheatWeight * priceVol * r.var95;
    
    const html = `
      <div class="card">
        <h3>Marginal VaR by Commodity</h3>
        <div class="results-grid">
          <div class="result-card">
            <h4>Corn</h4>
            <div class="value negative">${this.formatCurrency(cornMarginal)}</div>
            <div class="subtitle">${(cornWeight * 100).toFixed(1)}% of portfolio</div>
          </div>
          <div class="result-card">
            <h4>Soybeans</h4>
            <div class="value negative">${this.formatCurrency(soyMarginal)}</div>
            <div class="subtitle">${(soyWeight * 100).toFixed(1)}% of portfolio</div>
          </div>
          <div class="result-card">
            <h4>Wheat</h4>
            <div class="value negative">${this.formatCurrency(wheatMarginal)}</div>
            <div class="subtitle">${(wheatWeight * 100).toFixed(1)}% of portfolio</div>
          </div>
        </div>
      </div>
      
      <div class="card">
        <h3>Risk Factor Analysis</h3>
        <div class="heatmap-container">
          <table class="heatmap-table">
            <thead>
              <tr>
                <th>Factor</th>
                <th>Volatility</th>
                <th>Contribution to VaR</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Price Risk</strong></td>
                <td>${document.getElementById('price-vol').value}%</td>
                <td>${r.contributions.price.toFixed(1)}%</td>
                <td><span class="status status--${this.riskFactors.price.enabled ? 'success' : 'error'}">${this.riskFactors.price.enabled ? 'Active' : 'Inactive'}</span></td>
              </tr>
              <tr>
                <td><strong>Basis Risk</strong></td>
                <td>${document.getElementById('basis-vol').value}%</td>
                <td>${r.contributions.basis.toFixed(1)}%</td>
                <td><span class="status status--${this.riskFactors.basis.enabled ? 'success' : 'error'}">${this.riskFactors.basis.enabled ? 'Active' : 'Inactive'}</span></td>
              </tr>
              <tr>
                <td><strong>FX Risk</strong></td>
                <td>${document.getElementById('fx-vol').value}%</td>
                <td>${r.contributions.fx.toFixed(1)}%</td>
                <td><span class="status status--${this.riskFactors.fx.enabled ? 'success' : 'error'}">${this.riskFactors.fx.enabled ? 'Active' : 'Inactive'}</span></td>
              </tr>
              <tr>
                <td><strong>Freight Risk</strong></td>
                <td>${document.getElementById('freight-vol').value}%</td>
                <td>${r.contributions.freight.toFixed(1)}%</td>
                <td><span class="status status--${this.riskFactors.freight.enabled ? 'success' : 'error'}">${this.riskFactors.freight.enabled ? 'Active' : 'Inactive'}</span></td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
      
      <div class="card">
        <h3>Key Insights</h3>
        <div class="info-box">
          <p><strong>Dominant Risk Factor:</strong> ${this.getDominantFactor(r.contributions)}</p>
          <p><strong>Portfolio Diversification:</strong> ${this.assessDiversification()}</p>
          <p><strong>Tail Risk:</strong> ES/VaR ratio of ${(r.es975 / r.var95).toFixed(2)} indicates ${r.es975 / r.var95 > 1.3 ? 'significant' : 'moderate'} tail risk</p>
        </div>
      </div>
    `;
    
    decompDiv.innerHTML = html;
  }
  
  getDominantFactor(contributions) {
    const factors = [
      { name: 'Price Risk', value: contributions.price },
      { name: 'Basis Risk', value: contributions.basis },
      { name: 'FX Risk', value: contributions.fx },
      { name: 'Freight Risk', value: contributions.freight }
    ];
    const max = factors.reduce((prev, curr) => prev.value > curr.value ? prev : curr);
    return `${max.name} (${max.value.toFixed(1)}% contribution)`;
  }
  
  assessDiversification() {
    const cornPct = (this.portfolio.corn.qty * this.portfolio.corn.price) / 
      (this.portfolio.corn.qty * this.portfolio.corn.price + 
       this.portfolio.soybeans.qty * this.portfolio.soybeans.price + 
       this.portfolio.wheat.qty * this.portfolio.wheat.price);
    
    if (cornPct > 0.5 || cornPct < 0.2) return 'Consider rebalancing for better diversification';
    return 'Well-diversified across commodities';
  }
  
  runStressTest(index) {
    const scenarios = [
      { name: 'Moderate Downturn', price: -0.20, basis: 0.10, fx: 0.05, freight: 0.15 },
      { name: 'Severe Crisis', price: -0.40, basis: 0.20, fx: 0.10, freight: 0.30 },
      { name: 'Perfect Storm', price: -0.30, basis: 0.25, fx: 0.15, freight: 0.40 }
    ];
    
    const scenario = scenarios[index];
    const pnl = this.calculateStressPnL(scenario);
    
    document.getElementById(`stress-result-${index}`).textContent = this.formatCurrency(pnl);
    
    // Update chart
    this.updateStressChart();
  }
  
  runCustomStress() {
    const scenario = {
      name: 'Custom',
      price: parseFloat(document.getElementById('custom-price').value) / 100,
      basis: parseFloat(document.getElementById('custom-basis').value) / 100,
      fx: parseFloat(document.getElementById('custom-fx').value) / 100,
      freight: parseFloat(document.getElementById('custom-freight').value) / 100
    };
    
    const pnl = this.calculateStressPnL(scenario);
    document.getElementById('custom-stress-result').textContent = this.formatCurrency(pnl);
    
    this.updateStressChart();
  }
  
  calculateStressPnL(scenario) {
    let totalPnL = 0;
    const fxRate = parseFloat(document.getElementById('fx-rate').value);
    const freightCost = parseFloat(document.getElementById('freight-cost').value);
    
    const commodities = [
      { name: 'corn', data: this.portfolio.corn, basis: -0.05 },
      { name: 'soybeans', data: this.portfolio.soybeans, basis: -0.08 },
      { name: 'wheat', data: this.portfolio.wheat, basis: -0.06 }
    ];
    
    for (const commodity of commodities) {
      const { data, basis } = commodity;
      let newPrice = data.price * (1 + scenario.price);
      
      if (this.riskFactors.basis.enabled) {
        const basisChange = data.price * (basis + scenario.basis);
        newPrice += basisChange;
      }
      
      if (this.riskFactors.fx.enabled) {
        const newFxRate = fxRate * (1 + scenario.fx);
        newPrice = newPrice * newFxRate / fxRate;
      }
      
      if (this.riskFactors.freight.enabled) {
        const newFreight = freightCost * (1 + scenario.freight);
        newPrice -= newFreight;
      }
      
      totalPnL += data.qty * (newPrice - data.price);
    }
    
    return totalPnL;
  }
  
  updateStressChart() {
    const results = [];
    const labels = [];
    
    for (let i = 0; i < 3; i++) {
      const resultElem = document.getElementById(`stress-result-${i}`);
      if (resultElem.textContent) {
        const scenarioNames = ['Moderate Downturn', 'Severe Crisis', 'Perfect Storm'];
        labels.push(scenarioNames[i]);
        results.push(parseFloat(resultElem.textContent.replace(/[$,]/g, '')));
      }
    }
    
    const customResult = document.getElementById('custom-stress-result');
    if (customResult.textContent) {
      labels.push('Custom');
      results.push(parseFloat(customResult.textContent.replace(/[$,]/g, '')));
    }
    
    if (results.length === 0) return;
    
    const chartContainer = document.getElementById('stress-chart-container');
    chartContainer.style.display = 'block';
    
    const ctx = document.getElementById('stress-chart');
    if (this.charts.stress) this.charts.stress.destroy();
    
    this.charts.stress = new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label: 'Stress Test P&L',
          data: results,
          backgroundColor: '#B4413C',
          borderWidth: 0
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: { display: true, text: 'Stress Test Results' },
          legend: { display: false }
        },
        scales: {
          y: {
            title: { display: true, text: 'P&L ($)' },
            ticks: {
              callback: (value) => this.formatCurrency(value, true)
            }
          }
        }
      }
    });
  }
  
  // Mathematical utilities
  randomNormal() {
    // Box-Muller transform
    const u1 = Math.random();
    const u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }
  
  randomChiSquare(df) {
    // Sum of squared standard normals
    let sum = 0;
    for (let i = 0; i < df; i++) {
      const z = this.randomNormal();
      sum += z * z;
    }
    return sum;
  }
  
  choleskyDecomposition(matrix) {
    const n = matrix.length;
    const L = Array(n).fill(0).map(() => Array(n).fill(0));
    
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let sum = 0;
        for (let k = 0; k < j; k++) {
          sum += L[i][k] * L[j][k];
        }
        
        if (i === j) {
          L[i][j] = Math.sqrt(Math.max(0, matrix[i][i] - sum));
        } else {
          L[i][j] = (matrix[i][j] - sum) / (L[j][j] || 1);
        }
      }
    }
    
    return L;
  }
  
  multiplyMatrixVector(matrix, vector) {
    const result = [];
    for (let i = 0; i < matrix.length; i++) {
      let sum = 0;
      for (let j = 0; j < vector.length; j++) {
        sum += matrix[i][j] * vector[j];
      }
      result.push(sum);
    }
    return result;
  }
  
  formatCurrency(value, short = false) {
    const abs = Math.abs(value);
    const sign = value < 0 ? '-' : '';
    
    if (short && abs >= 1000000) {
      return `${sign}$${(abs / 1000000).toFixed(2)}M`;
    }
    
    return `${sign}$${abs.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`;
  }
}

// Initialize the application
const app = new CommodityRiskEngine();