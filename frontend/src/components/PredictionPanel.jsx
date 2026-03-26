export default function PredictionPanel({ prediction, loading }) {
    if (loading) {
        return (
            <div className="metric-card flex items-center justify-center h-72">
                <div className="spinner" />
            </div>
        );
    }

    if (!prediction) {
        return (
            <div className="metric-card flex items-center justify-center h-72 text-slate-500 text-sm">
                No model loaded
            </div>
        );
    }

    const { signal, probability_up, probability_down, current_price, training_metrics } = prediction;
    const probUp = (probability_up * 100).toFixed(1);
    const probDown = (probability_down * 100).toFixed(1);

    const signalClass = signal === 'BUY' ? 'buy' : signal === 'SELL' ? 'sell' : 'hold';
    const signalIcon = signal === 'BUY' ? '▲' : signal === 'SELL' ? '▼' : '●';

    return (
        <div className="metric-card space-y-5">
            {/* Header */}
            <div className="flex items-center justify-between">
                <h3 className="text-xs uppercase tracking-widest text-slate-500 font-medium">AI Prediction</h3>
                <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" title="Model Active" />
            </div>

            {/* Signal Badge */}
            <div className="text-center py-2">
                <div className={`signal-badge ${signalClass} text-lg mx-auto`}>
                    {signalIcon} {signal}
                </div>
            </div>

            {/* Price */}
            <div className="text-center">
                <div className="text-2xl font-bold tracking-tight">{current_price?.toFixed(current_price > 100 ? 2 : 4)}</div>
                <div className="text-xs text-slate-500 mt-1">Current Price</div>
            </div>

            {/* Probability Bars */}
            <div className="space-y-3">
                <div>
                    <div className="flex justify-between text-xs mb-1">
                        <span className="text-green-400">▲ Price Up</span>
                        <span className="text-green-400 font-semibold">{probUp}%</span>
                    </div>
                    <div className="prob-bar-bg">
                        <div className="prob-bar-fill bg-gradient-to-r from-green-500 to-emerald-400" style={{ width: `${probUp}%` }} />
                    </div>
                </div>
                <div>
                    <div className="flex justify-between text-xs mb-1">
                        <span className="text-red-400">▼ Price Down</span>
                        <span className="text-red-400 font-semibold">{probDown}%</span>
                    </div>
                    <div className="prob-bar-bg">
                        <div className="prob-bar-fill bg-gradient-to-r from-red-500 to-rose-400" style={{ width: `${probDown}%` }} />
                    </div>
                </div>
            </div>

            {/* Model Metrics */}
            {training_metrics && (
                <div className="border-t border-slate-700/50 pt-3 mt-3">
                    <div className="text-[10px] uppercase tracking-widest text-slate-600 mb-2">Model Metrics</div>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                        <div className="flex justify-between"><span className="text-slate-500">Accuracy</span><span>{(training_metrics.accuracy * 100).toFixed(1)}%</span></div>
                        <div className="flex justify-between"><span className="text-slate-500">F1 Score</span><span>{(training_metrics.f1_score * 100).toFixed(1)}%</span></div>
                        <div className="flex justify-between"><span className="text-slate-500">Precision</span><span>{(training_metrics.precision * 100).toFixed(1)}%</span></div>
                        <div className="flex justify-between"><span className="text-slate-500">ROC AUC</span><span>{(training_metrics.roc_auc * 100).toFixed(1)}%</span></div>
                    </div>
                </div>
            )}
        </div>
    );
}
