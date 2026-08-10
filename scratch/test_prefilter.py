import sys, os, asyncio
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import MetaTrader5 as mt5
from mt5.connection import MT5ConnectionManager
from mt5.data import MT5DataReader
from core.analysis_engine import MarketAnalysisEngine
from app_config.settings import settings

def test_prefilter_reason(m5_analysis, min_adx=15.0, require_rising=False):
    indicators = (m5_analysis or {}).get("indicators", {})
    range_setup = (m5_analysis or {}).get("market_structure", {}).get("range_reversion", {}) or {}
    range_can_continue = bool(range_setup.get("eligible"))
    try:
        adx = float(indicators.get("adx_14", 0.0) or 0.0)
    except Exception:
        adx = 0.0
    if adx < min_adx and not range_can_continue:
        return f"REJECTED: ADX {adx:.1f} < {min_adx:.1f}"
    if require_rising and not range_can_continue:
        adx_delta = float(indicators.get("adx_delta", 0.0) or 0.0)
        if adx_delta < -0.5:
            return f"REJECTED: ADX falling ({adx_delta:+.2f})"
    return ""

async def main():
    conn = MT5ConnectionManager()
    await conn.initialize()
    reader = MT5DataReader(conn)
    analyzer = MarketAnalysisEngine()
    symbols = ['USDJPY','USDCAD','CADJPY','EURUSD','GBPUSD','AUDUSD','NZDUSD','USDCHF','EURJPY','GBPJPY','EURGBP','AUDCAD']
    
    print("--- Comparing Default (ADX>=20, Rising=True) vs New (ADX>=15, Rising=False) ---")
    for s in symbols:
        df = await reader.get_ohlcv(s, "M5", 100)
        if df is not None:
            analysis = analyzer.analyze(s, "M5", df)
            r_old = test_prefilter_reason(analysis, min_adx=20.0, require_rising=True)
            r_new = test_prefilter_reason(analysis, min_adx=15.0, require_rising=False)
            ind = analysis.get("indicators", {})
            adx = ind.get("adx_14", 0.0)
            print(f"{s:8s} ADX={adx:4.1f} | Old: {'PASS' if not r_old else r_old[:30]:30s} | New: {'PASS' if not r_new else r_new[:30]:30s}")
    await conn.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
