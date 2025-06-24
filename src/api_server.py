from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool

from src.hedgefund.core import HedgeFundRunner, default_dates
from src.hedgefund.portfolio import create_portfolio

app = FastAPI(title="AI Hedge Fund API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TradeRequest(BaseModel):
    tickers: list[str]
    start_date: str | None = None
    end_date: str | None = None
    initial_cash: float = 100000.0
    margin_requirement: float = 0.0
    show_reasoning: bool = False
    analysts: list[str] | None = None
    model_name: str = "gpt-4o"
    model_provider: str = "OpenAI"


@app.post("/trade")
async def trade(req: TradeRequest):
    start_date, end_date = default_dates(req.start_date, req.end_date)
    portfolio = create_portfolio(req.initial_cash, req.margin_requirement, req.tickers)
    runner = HedgeFundRunner(req.analysts)
    try:
        result = await run_in_threadpool(
            runner.run,
            req.tickers,
            start_date,
            end_date,
            portfolio,
            req.show_reasoning,
            req.model_name,
            req.model_provider,
        )
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


class BacktestRequest(BaseModel):
    tickers: list[str]
    start_date: str | None = None
    end_date: str | None = None
    initial_cash: float = 100000.0
    margin_requirement: float = 0.0
    analysts: list[str] | None = None
    model_name: str = "gpt-4o"
    model_provider: str = "OpenAI"


@app.post("/backtest")
async def backtest(req: BacktestRequest):
    from src.backtester import Backtester

    start_date, end_date = default_dates(req.start_date, req.end_date)
    runner = HedgeFundRunner(req.analysts)
    backtester = Backtester(
        agent=runner.run,
        tickers=req.tickers,
        start_date=start_date,
        end_date=end_date,
        initial_capital=req.initial_cash,
        model_name=req.model_name,
        model_provider=req.model_provider,
        selected_analysts=req.analysts or [],
        initial_margin_requirement=req.margin_requirement,
    )
    result = await run_in_threadpool(backtester.run_backtest)
    return result

