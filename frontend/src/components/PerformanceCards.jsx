export default function PerformanceCards({ metrics }) {
    if (!metrics) return null;

    const cards = [
        {
            label: 'Total Return',
            value: `${metrics.total_return_pct?.toFixed(2)}%`,
            delta: metrics.total_return_pct,
            format: 'pct',
        },
        {
            label: 'Sharpe Ratio',
            value: metrics.sharpe_ratio?.toFixed(3),
            delta: metrics.sharpe_ratio,
            format: 'num',
        },
        {
            label: 'Win Rate',
            value: `${metrics.win_rate_pct?.toFixed(1)}%`,
            delta: metrics.win_rate_pct - 50,
            format: 'pct',
        },
        {
            label: 'Max Drawdown',
            value: `${metrics.max_drawdown_pct?.toFixed(2)}%`,
            delta: -metrics.max_drawdown_pct,
            format: 'pct',
            invert: true,
        },
        {
            label: 'Sortino Ratio',
            value: metrics.sortino_ratio?.toFixed(3),
            delta: metrics.sortino_ratio,
            format: 'num',
        },
        {
            label: 'Profit Factor',
            value: metrics.profit_factor?.toFixed(3),
            delta: metrics.profit_factor - 1,
            format: 'num',
        },
        {
            label: 'Total Trades',
            value: metrics.total_trades,
            format: 'int',
        },
        {
            label: 'Final Equity',
            value: `$${metrics.final_equity?.toLocaleString(undefined, { maximumFractionDigits: 0 })}`,
            delta: metrics.total_return_pct,
            format: 'pct',
        },
    ];

    return (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {cards.map((card, i) => (
                <div key={i} className="metric-card fade-in" style={{ animationDelay: `${i * 50}ms` }}>
                    <div className="label">{card.label}</div>
                    <div className="value">{card.value}</div>
                    {card.delta !== undefined && card.format !== 'int' && (
                        <div className={`delta ${card.delta >= 0 ? 'positive' : 'negative'}`}>
                            {card.delta >= 0 ? '↑' : '↓'} {Math.abs(card.delta).toFixed(2)}{card.format === 'pct' ? '%' : ''}
                        </div>
                    )}
                </div>
            ))}
        </div>
    );
}
