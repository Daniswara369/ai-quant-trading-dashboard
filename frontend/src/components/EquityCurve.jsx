import Plot from 'react-plotly.js';

export default function EquityCurve({ data, symbol }) {
    if (!data || data.length === 0) {
        return <div className="flex items-center justify-center h-48 text-slate-500 text-sm">No equity curve data</div>;
    }

    const timestamps = data.map(d => d.timestamp);
    const equity = data.map(d => d.equity);
    const initial = equity[0] || 100000;

    const trace = {
        type: 'scatter',
        mode: 'lines',
        x: timestamps,
        y: equity,
        name: 'Equity',
        fill: 'tozeroy',
        line: { color: '#10b981', width: 2 },
        fillcolor: 'rgba(16, 185, 129, 0.08)',
    };

    const initialLine = {
        type: 'scatter',
        mode: 'lines',
        x: [timestamps[0], timestamps[timestamps.length - 1]],
        y: [initial, initial],
        name: `Initial: $${initial.toLocaleString()}`,
        line: { color: '#475569', width: 1, dash: 'dash' },
    };

    const layout = {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: '#94a3b8', family: 'Inter, sans-serif', size: 11 },
        margin: { l: 55, r: 20, t: 10, b: 35 },
        xaxis: {
            gridcolor: '#1e293b',
            showgrid: false,
            type: 'category',
            nticks: 6,
        },
        yaxis: {
            gridcolor: '#1e293b',
            tickprefix: '$',
            tickformat: ',.0f',
        },
        showlegend: true,
        legend: {
            orientation: 'h', y: 1.08, x: 0,
            font: { size: 10, color: '#64748b' },
            bgcolor: 'transparent',
        },
        hovermode: 'x unified',
    };

    return (
        <Plot
            data={[trace, initialLine]}
            layout={layout}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%', height: '280px' }}
            useResizeHandler
        />
    );
}
