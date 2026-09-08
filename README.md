# LLM Trading Terminal

## Laptop 24/7 profile

The MSI GF63 11UCX has an RTX 2050 Laptop GPU with 4 GB VRAM. A 9B Q6/Q8
model will spill into system RAM and is not an appropriate low-latency 24/7
profile for that machine. Use LM Studio's `qwen/qwen3.5-4b` model with the
Q4_K_M GGUF on a 4 GB machine, keep thinking disabled, and merge the non-secret
values from `.env.laptop.example` into the laptop's private `.env`. The local
provider accepts both the catalog ID `qwen/qwen3.5-4b` and LM Studio's shorter
loaded API identifier `qwen3.5-4b`.

The 4,096-token context is intentional: measured entry prompts are about 1,000
tokens and responses remain below the 220-token cap. Larger context does not
add market evidence, but it reserves more KV-cache memory and can increase
latency. Keep `LOCAL_LLM_STRUCTURED_OUTPUT=True`; LM Studio's global Structured
Output toggle may remain off because the application sends the schema itself.

Run exactly one armed engine per Pepperstone account. The workspace mutex is
local to one computer; simultaneously arming the desktop and laptop can create
duplicate orders and split risk/history state. During migration, stop the
desktop engine with no open positions, copy the private `.env` and
`trading_system.db`, start LM Studio and MT5 on the laptop, then launch
`run_unattended.ps1`. A second installation should remain disarmed standby.

Official hardware and model references:

- https://www.msi.com/Laptop/GF63-Thin-11UCX-s/Specification
- https://lmstudio.ai/models/qwen/qwen3.5-4b
- https://huggingface.co/Qwen/Qwen3.5-4B

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
  and H4** for confirmation. The unfinished M5 bar is deliberately excluded;
  it can change before close. Same-thesis re-entry waits for one wholly
  post-close M5 bar after a protected winner and two after a loss/flat close,
  and still requires a new structure, retest, or range-reversal trigger.
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
  broker-reported floating P/L, and manages profit in original-risk (R) units.
  The first broker profit floor requires both +$0.35 estimated net and +0.50R
  before protecting +$0.08. At +$0.60 and +0.75R it protects 35% of current
  net profit. At both +$1.15 and +0.60R, a hybrid tier protects the smaller of
  +$1.00 or +0.55R; after +1.25R it protects at least +0.70R using immutable
  initial risk (about $0.98 for a $1.40-risk trade). If the risk baseline is
  unavailable, +$1.15 to +$1.00 remains the bounded recovery fallback. Normal
  bot trades use a 1.00R trailing distance after +1.00R. Break-even remains at
  +0.75R, and stagnant setups retire after 12 M5 bars.
- Runs deterministic analysis across the active market universe, then sends
  only the configured number of top-ranked, capital-fit setups to the serial
  local-model queue (three on the desktop profile, two on the 4 GB laptop).
  This keeps later decisions inside the fresh-candle window.
- Records valid BUY/SELL signals rejected by confidence, risk, or shadow mode
  and evaluates their 60-minute broker-M1 outcomes. This telemetry is
  diagnostic only and never bypasses a gate or places an order.
- Keeps the normal entry-confidence floor at 70%. A 60-69% decision can proceed
  only as a bounded failed-thesis reversal: the latest opposite-direction trade
  must be a broker-confirmed loss, and fresh M5/M15 direction, structure,
  reversal-pattern, and strengthening-momentum checks must all agree. The
  planner, sizing, spread, margin, portfolio, and broker gates still apply.
- Shows a rolling 48-hour BUY/SELL funnel with candidate, rejection, execution,
  closed-P/L, and rejected-signal expectancy metrics. This makes directional
  skew visible without automatically weakening one side after a small sample.
- A modest M5 ADX decline may reach the model only when a fresh deterministic
  M5 trigger remains strong and M5/M15/H1 direction agrees. The complete risk
  gate repeats this check, so market discovery cannot authorize an order.
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
     remain deterministic and unchanged. With
     `DETERMINISTIC_ENTRY_FAST_PATH_ENABLED=true`, a uniquely directional,
     fresh, fully aligned M5/M15/H1/H4 setup uses the millisecond rules policy
     instead of waiting behind the local-model queue. Ambiguous setups still
     use the configured model or HOLD, and every downstream safety gate remains
     mandatory.
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
- Broker history reports every activated stop as a stop-loss event. The
  terminal labels a stop-triggered exit with positive net P/L as
  `PROTECTIVE_STOP`, distinguishing a successful profit floor or trailing stop
  from an initial losing `STOP_LOSS` without rewriting the broker record.
- Early profit locks use current cost-adjusted profit, not a past peak. The first two
  tiers require their USD and R thresholds simultaneously. The +$1.15 tier
  also requires +0.60R and caps its floor at +0.55R, so it can attempt +$1.00
  without becoming oversized for a small-risk trade. A mature +1.25R trade
  receives a scale-aware +0.70R minimum floor. If a restart cannot reconstruct
  R, the +$1.15 to +$1.00 rule remains the bounded fallback.
  Successful tiers are remembered per ticket so later tiers can upgrade the
  broker stop without repeating the same modification every poll. An ordinary
  giveback cannot market-close a position with a valid risk baseline until its
  peak reaches 1R by default; completed-M1 adverse-structure management and the
  broker stop remain active below that threshold.
  Mature **net-profit retention** additionally arms at an observed $1.50 net
  peak and 1.00R of initial monetary risk (when available). It requests at least
  a $1.00 floor, ratcheting to 65% of the net peak and 75% once the net peak
  reaches 2.00R. For $1.40 initial risk, $1.65 **net** requests a $1.07 floor;
  $2.80 net requests $2.10. It does not cap a continuing winner's profit or
  extend the technical TP. Tighter retention can also exit a recoverable pullback.
  The desired floor never decreases at unchanged size and survives restarts
  in account/ticket-scoped storage. Partial closes scale it to remaining size;
  adding volume resets the monetary baseline without loosening the broker SL.
  Net estimates include configured round-turn costs and current swap. The
  executor rounds toward protection on the broker tick grid and verifies the
  projected net P/L before requesting the SL. The UI separately reports the
  accepted broker floor estimate and desired net-retention floor. If net falls
  through the desired floor, the existing `PROFIT_GIVEBACK_ENABLED` switch also
  enables the software `PROFIT_RETENTION` exit, even if broker stop/freeze
  restrictions prevented the SL upgrade. This fallback requires the application
  and broker connection to remain running; gaps, slippage, unobserved peaks and
  differences between estimated and actual costs mean no realized $1 guarantee.
  Configure `PROFIT_RETENTION_ENABLED`, `PROFIT_RETENTION_TRIGGER_USD`,
  `PROFIT_RETENTION_TRIGGER_R`, `PROFIT_RETENTION_KEEP_FRACTION`,
  `PROFIT_RETENTION_MATURE_TRIGGER_R`, `PROFIT_RETENTION_MATURE_KEEP_FRACTION`,
  and `PROFIT_RETENTION_MIN_HEADROOM_USD` in `.env`, then restart the process.
  `PROFIT_LOCK_ENABLED=false` disables this additional retention layer too.
  Application-led closes also persist their exact strategy cause separately
  from MT5's generic `EXPERT` label, so broker reconciliation and restarts do
  not erase `PROFIT_GIVEBACK`, reversal, stagnation, model, or operator exits.
  Those exits are replayed diagnostically at 15/30/60 minutes against the
  original broker SL/TP. The UI reports hold-minus-actual R results, but this
  evidence never changes live rules automatically.
- An exhausted same-candle M5 BOS/breakout must also have a verified retest, a
  directional candle pattern, or an actual H1/H4 structure event. Higher-
  timeframe direction labels alone cannot authorize that chased setup.
- The normal anti-chase limit remains 1.50 ATR. A 1.50-2.00 ATR continuation
  can proceed only with at least 85% confidence, a fresh BOS/retest/confirmed
  CHoCH, exact M5/M15/H1/H4 alignment, and independently strong momentum. This
  exception does not apply to breakout-only signals and does not bypass spread,
  shock, R:R, re-entry, sizing, portfolio-risk, margin, or broker checks.
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

The local provider accepts any chat LLM served by LM Studio, with no publisher,
model-family, or quantization allow-list. Set `LLM_PROVIDER=local`, copy the
model's exact API identifier to `LOCAL_LLM_MODEL`, and keep
`LOCAL_LLM_REQUIRED_QUANTIZATION=AUTO`. For example, all of these are valid:

- `prism-ml/bonsai-27b`
- `qwen3.5-4b@q4_k_s`
- `qwen3.5-4b@iq4_xs`
- Any other installed chat model's catalog key or custom loaded instance ID.

Use literal `@` and `_` characters in `.env`, without Markdown backslashes.
An explicit `@variant` selects that variant only. An unsuffixed name can resolve
to its loaded variant; ambiguous names report an error instead of selecting an
unrelated model. `AUTO` accepts all quantizations, including models whose runtime
does not report one. Enter an exact quantization name in settings or `.env` to
enforce a pin.

Load the selected model in LM Studio with at least `LOCAL_LLM_CONTEXT_SIZE`
tokens and enough parallel slots for `LLM_MAX_CONCURRENCY`. Restart the terminal
after changing `.env`. Test it independently of the trading engine with:

```powershell
python -m llm.check
python -m llm.check --model "qwen3.5-4b@iq4_xs"
```

This sends a HOLD-only readiness request and exits with code 0 when successful.
It does not load models or place trades. Embedding models cannot serve chat
decisions. Model support does not guarantee valid JSON or adequate speed on
every machine: loading, context, timeout, and decision-validation checks still
apply. A runtime rejecting an optional request field gets a bounded compatibility
retry; unsupported structured output falls back to a JSON instruction. Invalid
or truncated output remains a failed decision.

For an RTX 3060 12 GB, Qwen3.5 9B Q6 remains an optional higher-capacity local
profile; Q8 consumes more VRAM and does not improve deterministic price, risk,
or broker validation. The active low-latency profile now uses Qwen3.5 4B. For
the 4 GB RTX 2050 laptop, use its Q4_K_M build as described above. A model that
spills heavily into system RAM can let a decision become stale before execution.

- The provider uses LM Studio's advertised reasoning capabilities to turn
  thinking off when available, or request the lowest advertised effort.
  Reasoning-only models may need a larger `LOCAL_LLM_MAX_TOKENS` than the default
  220 (up to 32768 can be configured), plus enough loaded context. The engine's
  existing time and freshness limits still apply.
- Turn LM Studio's global **Structured Output** toggle off if its schema box is
  empty. Keep `LOCAL_LLM_STRUCTURED_OUTPUT=True`; the application sends its own
  strict schema with each request.
- Temperature is `0.0` for repeatability.

## Opportunity scanning and broker-wide discovery

Discovery and entry now share completed M5/M15/H1/H4 analysis. Eligible range
reversals and verified pullback resumptions join fresh BOS/CHoCH and breakout
setups. Retests are reconstructed from adjacent closed M5 candles, even for a
newly discovered market; session gaps do not create retests.

Verified, affordable setups receive active-watch priority before markets still
waiting for a trigger. The existing deterministic structure/exhaustion checks
run before model admission, so an impossible setup does not consume a model
slot. Remaining candidates can use unused slots even if an older ranking put
them below the model cutoff. Actual confidence, exposure, spread, margin,
freshness, broker permissions, and final execution checks are unchanged. The
60% failed-thesis exception is still conditional, not a general confidence floor.

Optional `.env` settings (restart required):

```dotenv
DYNAMIC_MARKET_SELECTION_ENABLED=true
BROKER_MARKET_DISCOVERY_ENABLED=true
BROKER_MARKET_GROUP=*
BROKER_MARKET_BATCH_SIZE=12
BROKER_MARKET_REFRESH_SECONDS=300
```

Configured and currently selected markets are revisited each selection cycle.
Up to 12 additional broker instruments rotate into each assessment batch;
visible Market Watch contracts are ordered first on the initial pass. The
broker catalog refreshes every 300 seconds. This bounds analysis load: it does
not scan thousands of instruments every minute or increase the per-bar model
budget. Large catalogs take multiple cycles to cover. Use `BROKER_MARKET_GROUP`
to narrow coverage with the broker's symbol-name wildcard filters, as described
in [MT5 symbols_get](https://www.mql5.com/en/docs/python_metatrader5/mt5symbolsget_py).
The existing configured weekend-only universe remains in effect.

Discovery excludes custom, disabled, close-only, ambiguous-name and unusable
contracts, including contracts without market orders and broker SL/TP support.
Exact broker symbol casing and broker pip metadata are retained across market
data and execution. A listed contract can still be unavailable because of its
session, missing history, spread, account size, or risk limits. Dashboard market
cards show setup readiness, viable directions, and technical rejection reasons
separately from affordability. These changes increase coverage and review
efficiency, not guaranteed fills, win rate, or profitability.

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
