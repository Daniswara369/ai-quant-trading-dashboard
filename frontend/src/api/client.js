import axios from 'axios';

// Auto-format the API URL to guarantee it ends with /api to match FastAPI routes
let base = import.meta.env.VITE_API_URL || '/api';
if (base !== '/api' && !base.endsWith('/api')) {
    // Strip trailing slash if it exists before appending /api
    base = base.replace(/\/$/, '') + '/api';
}

const API = axios.create({
    baseURL: base,
    timeout: 30000,
});

export const fetchAssets = () => API.get('/assets');

export const fetchMarketData = (symbol, timeframe = '1h', limit = 200) =>
    API.get('/market-data', { params: { symbol, timeframe, limit } });

export const fetchSignals = (symbol, timeframe = '1h', model_type = 'xgboost', limit = 200) =>
    API.get('/signals', { params: { symbol, timeframe, model_type, limit } });

export const fetchPrediction = (symbol, timeframe = '1h', model_type = 'xgboost') =>
    API.get('/prediction', { params: { symbol, timeframe, model_type } });

export const fetchAllSignals = (timeframe = '1h', model_type = 'xgboost') =>
    API.get('/all-signals', { params: { timeframe, model_type } });

export const fetchBacktestResults = (symbol, timeframe = '1h') =>
    API.get('/backtest-results', { params: { symbol, timeframe } });

export const fetchEquityCurve = (symbol, timeframe = '1h') =>
    API.get('/equity-curve', { params: { symbol, timeframe } });

export const refreshData = (symbol, timeframe = '1h') =>
    API.post('/refresh-data', null, { params: { symbol, timeframe } });

export const refreshSignal = (symbol, timeframe = '1h', model_type = 'xgboost') =>
    API.post('/refresh-signal', null, { params: { symbol, timeframe, model_type } });

export default API;
