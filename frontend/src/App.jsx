import { useState, useEffect, useCallback } from 'react';
import TopNav from './components/TopNav';
import CandlestickChart from './components/CandlestickChart';
import PredictionPanel from './components/PredictionPanel';
import PerformanceCards from './components/PerformanceCards';
import EquityCurve from './components/EquityCurve';
import AssetOverview from './components/AssetOverview';
import AgentThoughts from './components/AgentThoughts';
import {
  fetchMarketData,
  fetchSignals,
  fetchPrediction,
  fetchAllSignals,
  fetchBacktestResults,
  fetchEquityCurve,
  refreshData as apiRefreshData,
  refreshSignal as apiRefreshSignal,
} from './api/client';

export default function App() {
  const [symbol, setSymbol] = useState('BTCUSDT');
  const [timeframe, setTimeframe] = useState('1h');
  const [showSignals, setShowSignals] = useState(true);
  const [loading, setLoading] = useState(false);

  // Data states
  const [chartData, setChartData] = useState([]);
  const [signalsData, setSignalsData] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [allAssets, setAllAssets] = useState([]);
  const [backtestMetrics, setBacktestMetrics] = useState(null);
  const [equityCurve, setEquityCurve] = useState([]);
  const [error, setError] = useState(null);

  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [marketRes, signalRes, predRes, allSigRes, btRes, eqRes] = await Promise.allSettled([
        fetchMarketData(symbol, timeframe, 200),
        fetchSignals(symbol, timeframe, 'xgboost', 200),
        fetchPrediction(symbol, timeframe, 'xgboost'),
        fetchAllSignals(timeframe, 'xgboost'),
        fetchBacktestResults(symbol, timeframe),
        fetchEquityCurve(symbol, timeframe),
      ]);

      if (marketRes.status === 'fulfilled') setChartData(marketRes.value.data.data);
      if (signalRes.status === 'fulfilled') setSignalsData(signalRes.value.data.signals);
      if (predRes.status === 'fulfilled') setPrediction(predRes.value.data);
      if (allSigRes.status === 'fulfilled') setAllAssets(allSigRes.value.data.assets);
      if (btRes.status === 'fulfilled') setBacktestMetrics(btRes.value.data.metrics);
      if (eqRes.status === 'fulfilled') setEquityCurve(eqRes.value.data.equity_curve);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [symbol, timeframe]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleRefreshData = async () => {
    setLoading(true);
    try {
      await apiRefreshData(symbol, timeframe);
      await loadData();
    } finally {
      setLoading(false);
    }
  };

  const handleRefreshSignal = async () => {
    setLoading(true);
    try {
      const res = await apiRefreshSignal(symbol, timeframe, 'xgboost');
      setPrediction(prev => ({
        ...prev,
        signal: res.data.signal,
        probability_up: res.data.probability_up,
        probability_down: res.data.probability_down,
        current_price: res.data.current_price,
      }));
      await loadData();
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen" style={{ background: 'var(--bg-primary)' }}>
      <TopNav
        symbol={symbol}
        timeframe={timeframe}
        onSymbolChange={setSymbol}
        onTimeframeChange={setTimeframe}
        onRefreshData={handleRefreshData}
        onRefreshSignal={handleRefreshSignal}
        loading={loading}
      />

      {/* Error Banner */}
      {error && (
        <div className="max-w-[1600px] mx-auto px-4 mt-3">
          <div className="bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-2 text-red-400 text-sm">
            {error}
          </div>
        </div>
      )}

      <main className="max-w-[1600px] mx-auto px-4 py-4 space-y-4">

        {/* Row 1: Chart + Prediction Panel */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
          {/* Chart — 3/4 width */}
          <div className="lg:col-span-3 metric-card p-2">
            <div className="flex items-center justify-between px-3 py-2">
              <div className="flex items-center gap-4">
                <h2 className="text-sm font-semibold text-slate-300">
                  {symbol} <span className="text-slate-600">•</span> <span className="text-slate-500">{timeframe}</span>
                </h2>
                <button 
                  onClick={() => setShowSignals(!showSignals)}
                  className={`px-3 py-1 text-[10px] font-bold tracking-wider uppercase rounded-full transition-all duration-300 border ${
                    showSignals 
                    ? 'bg-blue-500/20 text-blue-400 border-blue-500/30 hover:bg-blue-500/40' 
                    : 'bg-slate-800/80 text-slate-500 border-slate-700/80 hover:bg-slate-700 hover:text-slate-400'
                  }`}
                >
                  <div className={`w-1.5 h-1.5 rounded-full ${showSignals ? 'bg-blue-400 animate-pulse' : 'bg-slate-600'}`} />
                  {showSignals ? 'Signals ON' : 'Signals OFF'}
                </button>
              </div>
              {loading && <div className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }} />}
            </div>
            <CandlestickChart data={chartData} signals={signalsData} symbol={symbol} timeframe={timeframe} showSignals={showSignals} />
          </div>

          {/* Prediction — 1/4 width */}
          <div className="lg:col-span-1">
            <PredictionPanel prediction={prediction} loading={loading && !prediction} />
          </div>
        </div>

        {/* Row 2: Agent Committee Thoughts */}
        <div>
          <AgentThoughts symbol={symbol} timeframe={timeframe} />
        </div>

        {/* Row 3: Performance Metrics */}
        <div>
          <h2 className="text-xs uppercase tracking-widest text-slate-500 font-medium mb-3 px-1">
            Strategy Performance
          </h2>
          <PerformanceCards metrics={backtestMetrics} />
        </div>

        {/* Row 4: Equity Curve */}
        <div className="metric-card p-3">
          <h2 className="text-xs uppercase tracking-widest text-slate-500 font-medium mb-2">Equity Curve</h2>
          <EquityCurve data={equityCurve} symbol={symbol} />
        </div>

        {/* Row 5: Multi-Asset Overview */}
        <div>
          <h2 className="text-xs uppercase tracking-widest text-slate-500 font-medium mb-3 px-1">
            Multi-Asset Overview
          </h2>
          <AssetOverview assets={allAssets} onSelectAsset={setSymbol} currentSymbol={symbol} />
        </div>

        {/* Footer */}
        <div className="text-center py-6 text-xs text-slate-600">
          Quantryst AI Trading Platform v2.0 — For educational and research purposes only. Not financial advice.
        </div>
      </main>
    </div>
  );
}
