# LLM Trading Terminal

A loopback-only trading workstation for Pepperstone MetaTrader 5. It combines
deterministic price planning, broker-native risk and margin checks, position
protection, and a provider-aware decision service used only as one directional
input. The current profile uses a local LM Studio model; a no-cost deterministic
rules engine and OpenAI remain optional comparison providers.

This is a Python application using MetaQuotes' `MetaTrader5` integration. It is
not an MQL5 Expert Advisor. An MQL5 rewrite would change deployment, not the
strategy's expectancy.

> **No trading system can guarantee profit or safely promise to grow $15.**
> CFDs can lose the entire balance. This project prioritizes correct execution,
> bounded loss, transparent rejections, and evidence collection. Keep entries
> disarmed on a live account until the same build has passed a meaningful demo
> and walk-forward evaluation.

## What the engine does

- Reads live ticks and completed candles from the active Pepperstone MT5
  terminal.
- Starts a decision once per newly completed **M5** candle and uses **M15, H1,
  and H4** for confirmation.
- Sends indicators and market structure to the configured decision provider.
  The deterministic provider may return `BUY`, `SELL`, or `HOLD`; optional
  model providers may also propose an exit-only `CLOSE`. No provider can choose
  entry price, stop, target, volume, or call MT5 itself.
- Builds entry, stop loss, and take profit deterministically from the fresh
  broker quote, ATR, structure, minimum stop distance, spread, and configured
  reward-to-risk ratio.
- Sizes with MT5's account-currency profit calculation and rejects the setup if
  the broker minimum volume plus execution reserves is already over budget.
- Rechecks the active account, quote age, candle, spread, stop risk, total stop
  risk, margin, permissions, and broker `order_check` immediately before an
  order request.
- Reserves configured round-turn fees and worst-case allowed fill deviation in
  the same stop-risk budget. Configure the fee fields to the active Pepperstone
  account's actual commission schedule; zero means no extra commission reserve.
- Reconciles open positions and closed deals with MT5. History or position API
  failures block new entries instead of being treated as empty results.
- Verifies the resulting broker position after full or partial fills before the
  UI calls an operation complete; ambiguous state changes disarm entries.
- Persists each ticket's original one-R baseline so break-even/trailing rules
  cannot silently redefine risk after a restart.
- Reviews managed positions on every completed **M1** candle using M1/M5
  structure evidence while M5 remains the only entry timeframe.
- Polls broker positions every 0.5 seconds, persists each ticket's best
  broker-reported floating P/L, and manages profit in original-risk (R) units:
  the live profile can protect retained progress after +0.5R, move toward
  break-even, trail after +1R, and retire a stagnant setup after 12 M5 bars.
- Runs deterministic analysis across the active market universe, then sends
  only the top three capital-fit, prefilter-passing setups to the serial local
  model queue. This keeps later decisions inside the fresh-candle window.
- Records valid BUY/SELL signals rejected by confidence, risk, or shadow mode
  and evaluates their 60-minute broker-M1 outcomes. This telemetry is
  diagnostic only and never bypasses a gate or places an order.
- Manages break-even, trailing protection, optional profit/loss exits, and
  manual close or SL/TP updates from the terminal.

## Live-account safety

- Every process start is **disarmed**, even when `DRY_RUN=False`.
- Starting the engine enables monitoring and position protection only.
- Arming requires a confirmation phrase tied to the current DEMO or LIVE
  account.
- Account login/server/currency identity is revalidated before every mutation.
  Switching the account in MT5 automatically disarms entries.
- The web server only binds to `127.0.0.1`; a workspace lock prevents duplicate
  dashboard processes.
- A broker timeout is treated as ambiguous and is reconciled, not blindly
  retried as another order.

`DRY_RUN=True` simulates entries. `DRY_RUN=False` permits real order requests
only after explicit session arming and all gates pass.

## Micro-account constraints

The requested Forex bootstrap profile permits up to **10%** of current balance
per order because Pepperstone's 0.01-lot minimum often cannot fit a conventional
1% budget on a micro balance. The separate **$2.00** floating-loss close is only
an emergency backstop; the broker stop plus configured execution costs must
still fit the tighter 10% order cap before entry. This is high risk: at most two
positions may be open, aggregate stop risk is capped at 12%, and the UTC daily
loss ceiling remains $2.50. Broker minimum contract size,
spread, permitted slippage, configured fees, free margin, and the portfolio cap
still determine whether a symbol can fit.

The decision provider never controls position risk. Its output contains
direction, confidence, and reasoning; deterministic code applies the configured
risk envelope. This prevents any rule or model response from silently replacing
the user's risk budget.

The terminal's **Capital fit** panel recomputes affordability from current
Pepperstone contract specifications and quotes. The controlled bootstrap cap
does not guarantee an order: changing volatility can still produce a red
`UNAFFORDABLE` result, while spread, margin, daily-loss, confidence and execution
checks can independently reject a setup.

The three requested forex symbols remain the core watch list. On weekdays, the
adaptive selector ranks a bounded candidate universe by current capital fit,
spread headroom, margin headroom, ADX and transition state, then evaluates at
most eight affordable markets. Weekend candidates remain `ETHUSD,LTCUSD,XRPUSD`
while forex is closed. Selection is a feasibility/ranking step, not a directive
to trade: every setup still passes structure, risk, spread, margin and broker
checks. Pepperstone trading sessions and symbol availability still apply.

## Execution-cost configuration

The reserve is explicit because MT5 does not expose a universal future
commission estimate. Spread is already present in the live BID/ASK entry and is
not added twice.

- `MAX_ORDER_DEVIATION_POINTS=20` is both the submitted MT5 deviation and the
  adverse-fill distance reserved by sizing and final execution checks.
- `FX_ROUND_TURN_COST_USD_PER_LOT=7.0` is the default FX round-turn fee reserve.
  Set it to the actual schedule for the active Pepperstone account type.
- `CRYPTO_ROUND_TURN_COST_USD_PER_LOT=0.0` and
  `CFD_ROUND_TURN_COST_USD_PER_LOT=0.0` make no cross-asset fee assumption.
  Configure them when the account charges a per-lot fee.
- `FIXED_EXECUTION_COST_USD=0.0` can reserve an additional fixed round-turn fee
  per position.

If the final quote's worst permitted fill-to-stop loss plus configured costs
exceeds the risk budget, the order is rejected. Lower trade frequency on a $15
account is the intended result of that constraint.

The deterministic planner uses that same worst-fill and fee model when setting
take profit. The live configuration uses a 1.47 final execution gate and
`PLAN_NET_RR_BUFFER=0.00`; a positive buffer can be configured to keep planned
net R:R above the final gate when desired.

## Run

1. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

2. Open Pepperstone MetaTrader 5 and log into the intended account.
3. Configure one decision provider:

   - Recommended no-cost mode (the project default):

     ```dotenv
     LLM_PROVIDER=DETERMINISTIC
     ```

     This needs no API key, subscription, LM Studio process, or GPU inference.
     It evaluates completed-candle trend state, ADX, EMA, RSI, MACD,
     overextension, and CHoCH/BOS evidence directly, then passes any signal
     through the existing planner and all risk/broker gates.

   - Local optional: load `LOCAL_LLM_MODEL` in LM Studio and start its local
     OpenAI-compatible server, then set `LLM_PROVIDER=LOCAL`. On a 12 GB RTX
     3060, Q6 is preferred over Q8 for this multi-market workflow because it
     leaves more VRAM headroom and shortens the serial inference queue; final
     price, spread, structure, risk, margin, R:R, and `order_check` validation
     remain deterministic and unchanged.
   - OpenAI opt-in: configure these values in `.env` and restart the process:

     ```dotenv
     LLM_PROVIDER=OPENAI
     OPENAI_API_KEY=your-platform-api-key
     OPENAI_MODEL=gpt-5.6-terra
     OPENAI_REASONING_EFFORT=low
     LLM_MAX_CONCURRENCY=3
     ```

     Keep the key only in `.env`; the dashboard configuration endpoint
     deliberately refuses API-key updates.

4. Start the terminal:

   ```powershell
   python run_dashboard.py
   ```

5. Open <http://127.0.0.1:8080>.

The dashboard starts monitoring automatically when
`AUTO_START_MONITORING=True`; it does not arm new entries.

## Unattended operation

Unattended entry authorization is an explicit opt-in, not an environment-file
shortcut. In the terminal, select **Enable autonomy** and type the displayed
phrase, such as `ENABLE AUTONOMOUS LIVE`. The authorization is persisted only
for the exact Pepperstone company, server, login, and account mode that was
confirmed. It never transfers when the account in MT5 changes.

While enabled, the engine checks MT5 connectivity, trading permissions, broker
history, open-position stop-risk state, and the configured decision provider
every 15 seconds. It disarms entries during a failure and re-arms only after the
same account and all required services recover. Daily-loss, portfolio-risk,
margin, spread, confidence, stale-candle, and broker checks still apply.

For process-level recovery, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_unattended.ps1
```

The watchdog adopts an already-running service and supervises the API, decision
loop, broker-poll heartbeat, and independent position-protection loop. It
restarts only an identified process from this project after repeated internal
failures. An MT5/permission outage or unavailable LM Studio model is reported as
an external dependency failure and does not cause a Python restart loop. A
restart circuit limits repeated recovery attempts, and an occupied API port or
unrecognized process is never replaced with a duplicate engine. Events are
written to `watchdog.log`.

Windows must remain awake, Pepperstone MT5 must remain logged in with Algo
Trading enabled, and LM Studio must keep the configured model loaded. The
watchdog cannot restart those external applications. Manual **Disarm entries**
also clears persistent autonomy so the health supervisor cannot re-arm it.

This provides operational continuity, not evidence of profitability. Broker
SL/TP remains the final protection during a local process, model, or connection
outage.

## Scan cadence and observability

- `ANALYSIS_INTERVAL_SECONDS=15` is a polling and retry interval, not a request
  for four trades per minute.
- Each symbol gets at most one decision for a completed M5 candle.
- A managed position gets an exit-only review on each completed M1 candle.
  The model may close only the exact managed ticket and only when supplied M1
  or M5 opposing BOS, CHoCH, or breakout evidence passes deterministic
  validation.
- Tick collection runs independently of model inference. Broker ticks are
  checked every 150 ms and deterministic position protection every 0.5 seconds,
  so a slow model request cannot pause stop, profit-lock, or giveback handling.
- Profit peaks are account-and-ticket scoped and persisted. A restart therefore
  cannot erase the fact that a still-open trade previously crossed the
  configured profit-protection trigger.
- Once a position reaches the configured mid-profit threshold (0.5R by
  default), its broker-side profit floor retains the configured share of the
  cost-adjusted peak. The giveback guard uses the same threshold as a fallback
  without shortening the original stop before that progress exists.
- An exhausted same-candle M5 BOS/breakout must also have a verified retest, a
  directional candle pattern, or an actual H1/H4 structure event. Higher-
  timeframe direction labels alone cannot authorize that chased setup.
- Break-even protection is account-and-ticket scoped and becomes satisfied
  when the existing broker stop is already stronger, avoiding repeated MT5
  modification requests during the 0.5-second protection loop.
- The adaptive universe is refreshed every 60 seconds by default, outside the
  fresh-candle entry path. Unaffordable symbols can remain visible for discovery
  but are excluded before model inference, and a signal is discarded if its
  symbol falls out of the active ranking before execution.
- New-entry inference has a completed-candle deadline. A request that cannot
  obtain and finish its bounded model slot in time is cancelled instead of
  remaining in the queue and delaying later candles.
- Multiple symbols prepare data concurrently. Deterministic evaluation has no
  network or model queue. Local model calls default to one at a time; raise
  `LLM_MAX_CONCURRENCY` only after parallel schema probes pass without extra
  VRAM pressure. This RTX 3060 deployment is benchmarked and bounded at two.
  The OpenAI provider also uses bounded concurrency.
- A queued result is discarded if its analyzed M5 candle is no longer current.
- H4 disagreement remains a rejection by default. A controlled exception is
  enabled only for 85%+ decision confidence, M5/M15/H1 alignment with the action,
  M5 ADX of at least 34.5, at least 50% confluence, and RSI inside the 15-85
  exhaustion guard; the normal quality floor, spread, risk, margin, daily-loss,
  and broker checks still apply.
- A symbol card preserves its final decision or rejection reason until the next
  completed M5 candle begins instead of reverting immediately to `WAITING`.
- Every completed candle receives one shared transition label used by both
  Market Watch and the decision provider: `CONFIRMED_BULLISH/BEARISH`,
  `EARLY_BULLISH/BEARISH_REVERSAL`, `PULLBACK_IN_*_TREND`, or `NEUTRAL`.
  CHoCH/BOS events and the evidence behind that label are included directly in
  the multi-timeframe prompt. Market Watch no longer runs a conflicting second
  EMA-only trend calculation.
- A completed M5 range of at least 2.75 ATR or opening gap of at least 1.25 ATR
  pauses new entries for that candle. This reacts to visible shocks without
  pretending to predict an unannounced event before the broker reports it.
- Pepperstone's GMT+2/GMT+3 broker-clock timestamps are normalized to UTC before
  completed-candle validation.
- If both MT5 candle-history paths are populated but frozen while broker ticks
  remain current, the reader reconstructs only the missing completed candles
  from immutable MT5 tick history. It refuses incomplete, old, or more than
  12-hour gaps; unrecoverable data still blocks inference.
- An unexpected decision-loop or tick-loop exit marks the engine failed and
  disarms execution instead of leaving a false `RUNNING` status.
- The decision pipeline shows data collection, provider evaluation, deterministic
  planning, risk rejection, readiness, and order-check stages separately.
- The diagnostics panel reports GPU compute utilization and VRAM utilization as
  separate gauges.

## Agentic decision boundary

The decision service is intentionally agentic only at the analysis boundary:

```text
Pepperstone data -> indicators/structure -> provider decision
                 -> deterministic planner and risk manager
                 -> broker order_check -> MT5 execution -> reconciliation
```

- The default `rules-v2-adaptive` provider classifies qualified entries as trend
  continuation, pullback/resumption, or confirmed reversal. Reversals require a
  computed M5 CHoCH plus secondary structure evidence, M15 confirmation, three
  aligned momentum votes and adequate M5/M15 ADX. Pullback entries require the
  prior recorded M5 state to be a pullback and the current completed candle to
  resume the M15/H1 direction. H4 conflicts remain restricted to the configured
  strong-countertrend exception.
- Deterministic mode is repeatable, testable, and normally evaluates in
  milliseconds. It removes inference cost and delay; it does not prove higher
  win rate or profitability.
- OpenAI uses the Responses API with strict structured output, `store=false`,
  low default reasoning for the latency-sensitive M5 path, and no execution
  tools.
- A provider timeout, missing key, invalid JSON, or unavailable model fails
  closed. It does not silently switch providers or submit an order.
- The engine rechecks the completed candle, account identity, positions, quote,
  risk, margin, and broker response after inference.
- Every provider call—including HOLD and invalid responses—is written to the
  `decision_trace` SQLite table with provider, resolved model, latency, request
  ID, prompt hash, validation status, and response. This supplies the evidence
  needed for shadow/demo comparisons instead of assuming a provider change
  improved trading.

OpenAI implementation references:

- [Responses API and model guidance](https://developers.openai.com/api/docs/guides/latest-model)
- [OpenAI Agents SDK concepts](https://openai.github.io/openai-agents-python/)

## Market Watch quote layout

The requested custom display is preserved: the **red SELL tile displays the
higher quote** and the **blue BUY tile displays the lower quote**. That is a UI
presentation choice only. MT5 execution remains conventional: a BUY opens at
ASK and a SELL opens at BID. Spread calculations always use `ask - bid`.

## Optional LM Studio guidance

For an RTX 3060 12 GB, a 4-bit Qwen3.5 9B build is a practical low-latency
default. A much larger model that spills into system RAM can allow a decision
to become stale before execution; a deterministic, fast 9B response is more
useful here than a slow 26B response.

- Leave **Enable Thinking off** for the automated M5 loop.
- Turn LM Studio's global **Structured Output** toggle off if its schema box is
  empty. Keep `LOCAL_LLM_STRUCTURED_OUTPUT=True`; the application sends its own
  strict schema with each request.
- Temperature is `0.0` for repeatability.

## Before considering live arming

Use Pepperstone demo with the exact production build and contract symbols.
Collect at least 30 closed, cost-inclusive trades (preferably far more), then
evaluate net profit, profit factor, expectancy, maximum closed drawdown, spread
distribution, slippage, and walk-forward/out-of-sample results. The current
broker-reconciled strategy history is net negative and has not demonstrated a
positive out-of-sample expectancy; it does not yet justify assuming that
unattended live operation will be profitable.

Useful official references:

- [MetaTrader 5 Python integration](https://www.mql5.com/en/docs/python_metatrader5)
- [MT5 order checking](https://www.mql5.com/en/docs/python_metatrader5/mt5ordercheck_py)
- [Pepperstone trading hours](https://pepperstone.com/en-au/about-us/trading-hours)
