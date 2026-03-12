import { useState } from 'react';

const ASSETS = [
    { symbol: 'BTCUSDT', label: 'BTC/USDT', icon: '₿', market: 'crypto' },
    { symbol: 'ETHUSDT', label: 'ETH/USDT', icon: 'Ξ', market: 'crypto' },
    { symbol: 'AAPL', label: 'AAPL', icon: '', market: 'stock' },
    { symbol: 'MSFT', label: 'MSFT', icon: '', market: 'stock' },
    { symbol: 'TSLA', label: 'TSLA', icon: '', market: 'stock' },
    { symbol: 'EURUSD', label: 'EUR/USD', icon: '€', market: 'forex' },
    { symbol: 'GBPUSD', label: 'GBP/USD', icon: '£', market: 'forex' },
    { symbol: 'USDJPY', label: 'USD/JPY', icon: '¥', market: 'forex' },
];

const TIMEFRAMES = ['1m', '5m', '15m', '1h', '1d'];

export default function TopNav({ symbol, timeframe, onSymbolChange, onTimeframeChange, onRefreshData, onRefreshSignal, loading }) {
    const [showAssets, setShowAssets] = useState(false);

    return (
        <nav className="sticky top-0 z-50 border-b border-slate-800" style={{ background: 'rgba(10, 14, 23, 0.95)', backdropFilter: 'blur(12px)' }}>
            <div className="max-w-[1600px] mx-auto px-4 h-14 flex items-center justify-between gap-4">

                {/* Logo */}
                <div className="flex items-center gap-2 shrink-0">
                    <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-sm font-bold">Q</div>
                    <span className="text-sm font-semibold tracking-wide hidden sm:inline">Quantryst</span>
                </div>

                {/* Asset Selector */}
                <div className="relative">
                    <button
                        onClick={() => setShowAssets(!showAssets)}
                        className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700 hover:border-slate-600 text-sm font-medium transition-colors"
                    >
                        <span className="text-base">{ASSETS.find(a => a.symbol === symbol)?.icon}</span>
                        <span>{ASSETS.find(a => a.symbol === symbol)?.label || symbol}</span>
                        <svg className="w-3 h-3 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" /></svg>
                    </button>
                    {showAssets && (
                        <div className="absolute top-full left-0 mt-1 w-52 rounded-xl bg-slate-800 border border-slate-700 shadow-2xl overflow-hidden fade-in">
                            {['crypto', 'stock', 'forex'].map(market => (
                                <div key={market}>
                                    <div className="px-3 py-1.5 text-[10px] uppercase tracking-widest text-slate-500 bg-slate-900/50">{market}</div>
                                    {ASSETS.filter(a => a.market === market).map(a => (
                                        <button
                                            key={a.symbol}
                                            onClick={() => { onSymbolChange(a.symbol); setShowAssets(false); }}
                                            className={`w-full text-left px-3 py-2 text-sm hover:bg-slate-700/50 flex items-center gap-2 transition-colors ${a.symbol === symbol ? 'text-blue-400 bg-slate-700/30' : 'text-slate-300'}`}
                                        >
                                            <span>{a.icon}</span> {a.label}
                                        </button>
                                    ))}
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                {/* Timeframe Selector */}
                <div className="flex items-center gap-1 bg-slate-800/60 rounded-lg p-0.5 border border-slate-700/50">
                    {TIMEFRAMES.map(tf => (
                        <button
                            key={tf}
                            onClick={() => onTimeframeChange(tf)}
                            className={`px-2.5 py-1 rounded-md text-xs font-medium transition-all ${tf === timeframe
                                    ? 'bg-blue-600 text-white shadow-md'
                                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/50'
                                }`}
                        >
                            {tf}
                        </button>
                    ))}
                </div>

                {/* Spacer */}
                <div className="flex-1" />

                {/* Action Buttons */}
                <div className="flex items-center gap-2">
                    <button
                        onClick={onRefreshData}
                        disabled={loading}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-700 text-sm text-slate-300 hover:bg-slate-700 hover:text-white transition-colors disabled:opacity-40"
                    >
                        <svg className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" /></svg>
                        Refresh Data
                    </button>
                    <button
                        onClick={onRefreshSignal}
                        disabled={loading}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gradient-to-r from-blue-600 to-purple-600 text-sm font-medium text-white hover:from-blue-500 hover:to-purple-500 transition-all disabled:opacity-40 shadow-lg shadow-blue-900/20"
                    >
                        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
                        Refresh Signal
                    </button>
                </div>
            </div>
        </nav>
    );
}
