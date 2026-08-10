"""
main.py — Application entry point for the Local AI Trading System.

Wires up the modular components (MT5 connection, data reader, SQLite DB, risk manager,
order executor, provider-aware decision service) and instantiates the TradingEngine orchestrator.
Registers the dependencies used by the FastAPI lifespan in ui.dashboard.
"""
import uvicorn
from typing import Optional
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse
from trading_logger.logger import system_logger
from mt5.connection import MT5ConnectionManager
from mt5.data import MT5DataReader
from execution.executor import MT5OrderExecutor
from llm.client import LLMClient
from database.storage import TradingDatabase
from risk.manager import RiskManager
from core.engine import TradingEngine
from ui.dashboard import app  # import standard FastAPI instance
from ui.state import dashboard_state

# 1. Instantiate the modular component dependencies
conn_manager = MT5ConnectionManager()
data_reader = MT5DataReader(conn_manager)
executor = MT5OrderExecutor(conn_manager)
llm_client = LLMClient()
database = TradingDatabase()
risk_manager = RiskManager()

# 2. Inject dependencies into the Core Trading Engine orchestrator
engine = TradingEngine(
    connection_manager=conn_manager,
    data_reader=data_reader,
    executor=executor,
    llm_client=llm_client,
    database=database,
    risk_manager=risk_manager
)

# Bind the engine to the app state so the REST API endpoints can access it
app.state.engine = engine
app.state.database = database
app.state.connection_manager = conn_manager


class ArmRequest(BaseModel):
    confirmation: str = ""


class AutonomousRequest(BaseModel):
    confirmation: str = ""


class DailyLossResetRequest(BaseModel):
    confirmation: str = ""


class LossCooldownResetRequest(BaseModel):
    confirmation: str = ""


class LosingStreakResetRequest(BaseModel):
    confirmation: str = ""


class RejectedTradeRequest(BaseModel):
    confirmation: str = ""


class CloseRequest(BaseModel):
    percent: float = Field(default=100.0, ge=1.0, le=100.0)
    confirmation: str = ""


class ProtectRequest(BaseModel):
    stop_loss: Optional[float] = Field(default=None, gt=0)
    take_profit: Optional[float] = Field(default=None, gt=0)
    confirmation: str = ""

# 3. Add engine controls to the FastAPI REST API
@app.post("/api/start")
async def api_start_engine():
    """Start the trading bot loop."""
    success = await engine.start()
    if not success:
        return JSONResponse(
            {"status": "error", "message": "Failed to start trading engine."},
            status_code=503,
        )
    return {"status": "success", "message": "Trading engine started successfully."}

@app.post("/api/stop")
async def api_stop_engine():
    """Stop the trading bot loop."""
    await engine.stop()
    return {"status": "success", "message": "Trading engine stopped cleanly."}


@app.post("/api/entries/arm")
async def api_arm_entries(body: ArmRequest):
    ok, message = await engine.arm_entries(body.confirmation)
    return JSONResponse(
        {"status": "success" if ok else "error", "message": message},
        status_code=200 if ok else 400,
    )


@app.post("/api/entries/disarm")
async def api_disarm_entries():
    ok, message = await engine.operator_disarm_entries()
    return JSONResponse(
        {"status": "success" if ok else "error", "message": message},
        status_code=200 if ok else 500,
    )


@app.post("/api/autonomy/enable")
async def api_enable_autonomy(body: AutonomousRequest):
    ok, message = await engine.enable_autonomous_mode(body.confirmation)
    return JSONResponse(
        {"status": "success" if ok else "error", "message": message},
        status_code=200 if ok else 400,
    )


@app.post("/api/autonomy/disable")
async def api_disable_autonomy():
    ok, message = await engine.disable_autonomous_mode()
    return JSONResponse(
        {"status": "success" if ok else "error", "message": message},
        status_code=200 if ok else 500,
    )


@app.post("/api/risk/daily-loss/reset")
async def api_reset_daily_loss(body: DailyLossResetRequest):
    ok, message = await engine.reset_daily_loss_stop(body.confirmation)
    return JSONResponse(
        {"status": "success" if ok else "error", "message": message},
        status_code=200 if ok else 400,
    )


@app.post("/api/risk/loss-cooldown/reset")
async def api_reset_loss_cooldown(body: LossCooldownResetRequest):
    ok, message = await engine.reset_loss_cooldown(body.confirmation)
    return JSONResponse(
        {"status": "success" if ok else "error", "message": message},
        status_code=200 if ok else 400,
    )


@app.post("/api/risk/losing-streak/reset")
async def api_reset_losing_streak(body: LosingStreakResetRequest):
    ok, message = await engine.reset_losing_streak(body.confirmation)
    return JSONResponse(
        {"status": "success" if ok else "error", "message": message},
        status_code=200 if ok else 400,
    )


@app.get("/api/rejected/{symbol}/preview")
async def api_preview_rejected_trade(symbol: str):
    ok, payload = await engine.preview_rejected_trade(symbol)
    return JSONResponse(
        {"status": "success" if ok else "error", **payload},
        status_code=200 if ok else 409,
    )


@app.post("/api/rejected/{symbol}/open")
async def api_open_rejected_trade(symbol: str, body: RejectedTradeRequest):
    ok, message = await engine.open_rejected_trade(symbol, body.confirmation)
    return JSONResponse(
        {"status": "success" if ok else "error", "message": message},
        status_code=200 if ok else 409,
    )


@app.post("/api/positions/{ticket}/close")
async def api_close_position(ticket: int, body: CloseRequest):
    ok, message = await engine.close_position(ticket, body.percent, body.confirmation)
    return JSONResponse(
        {"status": "success" if ok else "error", "message": message},
        status_code=200 if ok else 400,
    )


@app.post("/api/positions/{ticket}/protect")
async def api_protect_position(ticket: int, body: ProtectRequest):
    ok, message = await engine.modify_position(
        ticket, body.stop_loss, body.take_profit, body.confirmation
    )
    return JSONResponse(
        {"status": "success" if ok else "error", "message": message},
        status_code=200 if ok else 400,
    )


@app.get("/api/llm/health")
async def api_llm_health():
    result = await llm_client.health_check()
    dashboard_state.update_automation(
        llm_online=bool(result.get("online") and result.get("available")),
        provider=str(result.get("provider", "local")),
        model=str(result.get("selected_model", "")),
    )
    return result

if __name__ == "__main__":
    system_logger.info("Starting local AI trading system uvicorn server...")
    uvicorn.run(app, host="127.0.0.1", port=8080)
