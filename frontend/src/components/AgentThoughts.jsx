import React, { useState, useEffect, useRef } from 'react';

export default function AgentThoughts({ symbol, timeframe }) {
  const [messages, setMessages] = useState([]);
  const [consensus, setConsensus] = useState(null);
  const [loading, setLoading] = useState(false);
  const wsRef = useRef(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    // Setup WebSocket
    const wsUrl = `ws://127.0.0.1:8000/ws/agent-thoughts`;
    wsRef.current = new WebSocket(wsUrl);

    wsRef.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'info') {
          setMessages(prev => [...prev, { type: 'info', text: data.message }]);
        } else if (data.type === 'consensus') {
           setConsensus(data.data);
           setMessages(prev => [...prev, { type: 'success', text: 'Analysis complete.' }]);
           setLoading(false);
        }
      } catch (err) {
        console.error("WS Parse error", err);
      }
    };

    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const runAnalysis = async () => {
    setLoading(true);
    setMessages([]);
    setConsensus(null);
    try {
      await fetch(`/api/agent-signal?symbol=${symbol}&timeframe=${timeframe}`, {
        method: 'POST'
      });
    } catch (err) {
      setMessages(prev => [...prev, { type: 'error', text: `Failed: ${err.message}` }]);
      setLoading(false);
    }
  };

  const executeMockTrade = async () => {
    if (!consensus) return;
    try {
      const res = await fetch(`/api/mock-trade?symbol=${symbol}&signal=${consensus.signal}&confidence=${consensus.confidence}&timeframe=${timeframe}`, {
        method: 'POST'
      });
      const data = await res.json();
      if (res.ok) {
        setMessages(prev => [...prev, { type: 'success', text: `Trade executed: ${JSON.stringify(data.execution)}` }]);
      } else {
        setMessages(prev => [...prev, { type: 'error', text: `Trade Reject: ${data.detail}` }]);
      }
    } catch (err) {
      setMessages(prev => [...prev, { type: 'error', text: `Trade Failed: ${err.message}` }]);
    }
  };

  const getAgentIcon = (name) => {
    switch (name) {
      case 'Analyst': return '🔧';
      case 'Sentiment Strategist': return '📰';
      case 'Risk Auditor': return '🧐';
      case 'Manager': return '👔';
      default: return '🤖';
    }
  };

  return (
    <div className="metric-card p-4 space-y-4">
      <div className="flex items-center justify-between border-b border-slate-700/50 pb-3">
        <h2 className="text-sm font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-emerald-400">
          Agent Committee Analysis
        </h2>
        <button 
          onClick={runAnalysis}
          disabled={loading}
          className="bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white text-xs px-3 py-1.5 rounded transition-all"
        >
          {loading ? 'Analyzing...' : 'Run Analysis'}
        </button>
      </div>

      <div className="flex flex-col md:flex-row gap-4">
        {/* Left Column: Debate Transcript & Status */}
        <div className="flex-1 space-y-2">
            <h3 className="text-xs text-slate-400 font-semibold uppercase tracking-wider">Live Thoughts</h3>
            <div className="h-48 overflow-y-auto bg-slate-900/50 rounded p-2 text-xs font-mono text-slate-300 space-y-1 border border-slate-800">
              {messages.length === 0 && <span className="text-slate-600">Waiting for trigger...</span>}
              {messages.map((m, i) => (
                <div key={i} className={m.type === 'error' ? 'text-red-400' : (m.type === 'success' ? 'text-emerald-400' : '')}>
                  {'>'} {m.text}
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>

            {consensus && consensus.regime && (
               <div className="mt-2 p-2 bg-slate-800/40 rounded border border-slate-700 text-xs">
                 <span className="text-slate-400 mr-2">Market Regime:</span>
                 <span className="text-indigo-400 font-bold">{consensus.regime.regime}</span>
                 <span className="text-slate-500 ml-2">({(consensus.regime.confidence * 100).toFixed(0)}% conf)</span>
               </div>
            )}
        </div>

        {/* Right Column: Agents Output */}
        <div className="flex-1 space-y-3">
            <h3 className="text-xs text-slate-400 font-semibold uppercase tracking-wider">Agent Signals</h3>
            
            <div className="space-y-2 max-h-48 overflow-y-auto pr-1">
              {!consensus && <div className="text-xs text-slate-500 italic">No consensus reached yet.</div>}
              
              {consensus && consensus.agent_outputs.map((agent, idx) => (
                <div key={idx} className="bg-slate-800/60 rounded p-2 border border-slate-700">
                  <div className="flex justify-between items-center mb-1">
                    <span className="font-bold text-xs text-slate-200">
                      {getAgentIcon(agent.agent_name)} {agent.agent_name}
                    </span>
                    <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
                        agent.signal === 'BUY' ? 'bg-emerald-500/20 text-emerald-400' : 
                        agent.signal === 'SELL' ? 'bg-red-500/20 text-red-400' : 
                        'bg-slate-500/20 text-slate-400'
                      }`}>
                      {agent.signal} ({(agent.confidence * 100).toFixed(0)}%)
                    </span>
                  </div>
                  <div className="text-[10px] text-slate-400 leading-tight">
                    {agent.reasoning}
                  </div>
                </div>
              ))}
            </div>
        </div>
      </div>

      {/* Footer: Final Consensus */}
      {consensus && (
        <div className="mt-4 pt-3 border-t border-slate-700/50 flex flex-col sm:flex-row justify-between items-center gap-4">
           <div className="flex items-center gap-3">
             <div className="text-2xl pt-1">👔</div>
             <div>
               <div className="text-xs text-slate-400 uppercase tracking-widest font-semibold flex items-center gap-2">
                 Manager Consensus
                 <span className={`text-xs px-2 py-0.5 rounded font-bold ${
                    consensus.signal === 'BUY' ? 'bg-emerald-500/20 text-emerald-400' : 
                    consensus.signal === 'SELL' ? 'bg-red-500/20 text-red-400' : 
                    'bg-slate-500/20 text-slate-400'
                  }`}>
                   {consensus.signal}
                 </span>
               </div>
               <div className="text-sm mt-1 text-slate-300">
                 Confidence: {(consensus.confidence * 100).toFixed(1)}%
               </div>
             </div>
           </div>

           <div className="flex-1 text-xs text-slate-400 border-l border-slate-700/50 pl-4 max-w-sm hidden sm:block">
              {consensus.reasoning}
           </div>

           <button 
             onClick={executeMockTrade}
             className="bg-emerald-600/20 hover:bg-emerald-600/40 text-emerald-400 border border-emerald-600/50 text-xs px-4 py-2 rounded transition-all font-bold uppercase tracking-wider whitespace-nowrap"
           >
             Mock Execute
           </button>
        </div>
      )}
    </div>
  );
}
