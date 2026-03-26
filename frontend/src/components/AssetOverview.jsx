export default function AssetOverview({ assets, onSelectAsset, currentSymbol }) {
    if (!assets || assets.length === 0) {
        return <div className="text-slate-500 text-sm text-center py-8">Loading asset overview...</div>;
    }

    return (
        <div className="overflow-x-auto rounded-xl border border-slate-800">
            <table className="data-table">
                <thead>
                    <tr>
                        <th>Asset</th>
                        <th>Price</th>
                        <th>Signal</th>
                        <th>P(Up)</th>
                        <th>Sharpe</th>
                        <th>Total Return</th>
                        <th>Win Rate</th>
                        <th>Max DD</th>
                    </tr>
                </thead>
                <tbody>
                    {assets.map((a, i) => {
                        const sigClass = a.signal === 'BUY' ? 'buy' : a.signal === 'SELL' ? 'sell' : 'hold';
                        const isActive = a.symbol === currentSymbol;
                        return (
                            <tr
                                key={a.symbol}
                                onClick={() => onSelectAsset(a.symbol)}
                                className={`cursor-pointer transition-colors ${isActive ? 'bg-slate-800/60' : ''}`}
                                style={{ animationDelay: `${i * 30}ms` }}
                            >
                                <td>
                                    <div className="flex items-center gap-2">
                                        <span className={`w-1.5 h-1.5 rounded-full ${isActive ? 'bg-blue-400' : 'bg-slate-600'}`} />
                                        <span className="font-medium text-slate-200">{a.symbol}</span>
                                        <span className="text-[10px] text-slate-600 uppercase">{a.market}</span>
                                    </div>
                                </td>
                                <td className="font-mono text-slate-200">
                                    {a.price > 0 ? a.price.toFixed(a.price > 100 ? 2 : 4) : '—'}
                                </td>
                                <td>
                                    <span className={`signal-badge ${sigClass}`} style={{ fontSize: '0.7rem', padding: '3px 10px' }}>
                                        {a.signal}
                                    </span>
                                </td>
                                <td className="font-mono">
                                    <span className={a.probability_up > 0.6 ? 'text-green-400' : a.probability_up < 0.4 ? 'text-red-400' : 'text-yellow-400'}>
                                        {(a.probability_up * 100).toFixed(1)}%
                                    </span>
                                </td>
                                <td className="font-mono">
                                    <span className={a.sharpe_ratio > 1 ? 'text-green-400' : a.sharpe_ratio > 0 ? 'text-yellow-400' : 'text-red-400'}>
                                        {a.sharpe_ratio?.toFixed(3)}
                                    </span>
                                </td>
                                <td className="font-mono">
                                    <span className={a.total_return_pct > 0 ? 'text-green-400' : 'text-red-400'}>
                                        {a.total_return_pct?.toFixed(2)}%
                                    </span>
                                </td>
                                <td className="font-mono">{a.win_rate_pct?.toFixed(1)}%</td>
                                <td className="font-mono text-red-400">{a.max_drawdown_pct?.toFixed(2)}%</td>
                            </tr>
                        );
                    })}
                </tbody>
            </table>
        </div>
    );
}
