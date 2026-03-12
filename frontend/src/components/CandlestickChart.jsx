import Plot from 'react-plotly.js';
import { useState } from 'react';

export default function CandlestickChart({ data, signals, symbol, timeframe, showSignals }) {
    if (!data || data.length === 0) {
        return <div className="flex items-center justify-center h-96 text-slate-500">No chart data available</div>;
    }

    const timestamps = data.map(d => d.timestamp);
    const opens = data.map(d => d.Open);
    const highs = data.map(d => d.High);
    const lows = data.map(d => d.Low);
    const closes = data.map(d => d.Close);
    const volumes = data.map(d => d.Volume || 0);

    const traces = [];

    // Candlestick
    traces.push({
        type: 'candlestick',
        x: timestamps,
        open: opens,
        high: highs,
        low: lows,
        close: closes,
        name: 'OHLC',
        increasing: { line: { color: '#10b981' }, fillcolor: '#10b981' },
        decreasing: { line: { color: '#ef4444' }, fillcolor: '#ef4444' },
        xaxis: 'x',
        yaxis: 'y',
    });

    // SMA overlay
    const smaColors = { SMA_10: '#fbbf24', SMA_20: '#fb923c', SMA_50: '#3b82f6' };
    for (const [key, color] of Object.entries(smaColors)) {
        if (data[0]?.[key] != null) {
            traces.push({
                type: 'scatter',
                mode: 'lines',
                x: timestamps,
                y: data.map(d => d[key]),
                name: key,
                line: { color, width: 1 },
                yaxis: 'y',
            });
        }
    }

    // EMA overlay
    ['EMA_12', 'EMA_26'].forEach((key, i) => {
        if (data[0]?.[key] != null) {
            traces.push({
                type: 'scatter',
                mode: 'lines',
                x: timestamps,
                y: data.map(d => d[key]),
                name: key,
                line: { color: i === 0 ? '#f472b6' : '#a78bfa', width: 1, dash: 'dot' },
                yaxis: 'y',
            });
        }
    });

    // Bollinger Bands
    if (data[0]?.BB_Upper != null) {
        traces.push({
            type: 'scatter', mode: 'lines',
            x: timestamps, y: data.map(d => d.BB_Upper),
            name: 'BB Upper', line: { color: 'rgba(59,130,246,0.3)', width: 1, dash: 'dot' }, yaxis: 'y',
        });
        traces.push({
            type: 'scatter', mode: 'lines',
            x: timestamps, y: data.map(d => d.BB_Lower),
            name: 'BB Lower', line: { color: 'rgba(59,130,246,0.3)', width: 1, dash: 'dot' },
            fill: 'tonexty', fillcolor: 'rgba(59,130,246,0.04)', yaxis: 'y',
        });
    }

    // Buy/Sell markers from signals
    if (showSignals && signals && signals.length > 0) {
        const buys = signals.filter(s => s.signal === 'BUY');
        const sells = signals.filter(s => s.signal === 'SELL');

        if (buys.length > 0) {
            traces.push({
                type: 'scatter', mode: 'markers',
                x: buys.map(s => s.timestamp), y: buys.map(s => s.price),
                name: 'BUY', marker: { symbol: 'triangle-up', size: 12, color: '#10b981', line: { width: 1, color: '#fff' } },
                yaxis: 'y',
            });
        }
        if (sells.length > 0) {
            traces.push({
                type: 'scatter', mode: 'markers',
                x: sells.map(s => s.timestamp), y: sells.map(s => s.price),
                name: 'SELL', marker: { symbol: 'triangle-down', size: 12, color: '#ef4444', line: { width: 1, color: '#fff' } },
                yaxis: 'y',
            });
        }
    }

    // Volume
    const volumeColors = closes.map((c, i) => c >= opens[i] ? 'rgba(16,185,129,0.3)' : 'rgba(239,68,68,0.3)');
    traces.push({
        type: 'bar',
        x: timestamps,
        y: volumes,
        name: 'Volume',
        marker: { color: volumeColors },
        yaxis: 'y2',
        xaxis: 'x',
    });

    const layout = {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: '#94a3b8', family: 'Inter, sans-serif', size: 11 },
        margin: { l: 55, r: 20, t: 10, b: 40 },
        xaxis: {
            rangeslider: { visible: false },
            gridcolor: '#1e293b',
            showgrid: false,
            type: 'category',
            nticks: 8,
        },
        yaxis: {
            domain: [0.22, 1],
            gridcolor: '#1e293b',
            gridwidth: 1,
            tickformat: '.2f',
        },
        yaxis2: {
            domain: [0, 0.18],
            gridcolor: '#1e293b',
            showticklabels: false,
        },
        showlegend: true,
        legend: {
            orientation: 'h', y: 1.06, x: 0,
            font: { size: 10, color: '#64748b' },
            bgcolor: 'transparent',
        },
        dragmode: 'zoom',
        hovermode: 'x unified',
    };

    return (
        <div className="w-full relative group">
            <Plot
                data={traces}
                layout={layout}
                config={{ displayModeBar: true, displaylogo: false, responsive: true, modeBarButtonsToRemove: ['select2d', 'lasso2d'] }}
                style={{ width: '100%', height: '520px' }}
                useResizeHandler
            />
        </div>
    );
}
