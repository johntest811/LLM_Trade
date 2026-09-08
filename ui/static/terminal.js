"use strict";

const app = {
  state: null,
  config: null,
  socket: null,
  reconnectAttempt: 0,
  reconnectTimer: null,
  reconnectCountdown: null,
  lastMessageAt: 0,
  fallbackFetchPending: false,
  lastFullStateAt: 0,
  lastSequence: 0,
  serverOffsetMs: 0,
  tickHistory: new Map(),
  tickKeys: new Set(),
  pending: new Set(),
  confirmResolve: null,
  toastCounter: 0,
  shuttingDown: false,
  managedPosition: null,
  rejectedPreview: null,
  renderFrame: null,
  pendingFullState: null,
  pendingLivePayload: null,
  renderHashes: new Map(),
  quoteCards: new Map(),
  watchInitialized: false,
};

const $ = (id) => document.getElementById(id);
const clamp = (value, min = 0, max = 100) => Math.max(min, Math.min(max, Number(value) || 0));
const finite = (value, fallback = 0) => Number.isFinite(Number(value)) ? Number(value) : fallback;

function node(tag, className = "", text = "") {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== "") element.textContent = String(text);
  return element;
}

function setText(id, value) {
  const element = $(id);
  if (element) element.textContent = value ?? "—";
}

function money(value, currency = "USD", precision = 2) {
  const number = Number(value);
  if (!Number.isFinite(number)) return `— ${currency}`;
  return `${number.toFixed(precision)} ${currency}`;
}

function signedMoney(value, currency = "USD") {
  const number = finite(value);
  return `${number > 0 ? "+" : ""}${number.toFixed(2)} ${currency}`;
}

function price(value) {
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0) return "—";
  return number.toFixed(number < 10 ? 5 : number < 1000 ? 3 : 2);
}

function colorValue(element, value) {
  if (!element) return;
  element.classList.remove("positive", "negative", "muted");
  const number = Number(value);
  element.classList.add(!Number.isFinite(number) || number === 0 ? "muted" : number > 0 ? "positive" : "negative");
}

function normalizeConfidence(value) {
  const number = finite(value);
  return clamp(number <= 1 ? number * 100 : number);
}

function parseDate(value) {
  if (!value || value === "—") return null;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}

function formatTimestamp(value, options = {}) {
  const date = parseDate(value);
  if (!date) return String(value || "—");
  return new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: options.seconds === false ? undefined : "2-digit",
    hour12: false,
  }).format(date);
}

function ageSeconds(value) {
  const date = parseDate(value);
  if (!date) return null;
  return Math.max(0, Math.floor((serverNow() - date.getTime()) / 1000));
}

function ageLabel(value) {
  const seconds = ageSeconds(value);
  if (seconds === null) return "time —";
  if (seconds < 2) return "now";
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  return `${Math.floor(seconds / 3600)}h ago`;
}

function serverNow() {
  return Date.now() + app.serverOffsetMs;
}

function syncServerClock(serverTime) {
  const parsed = parseDate(serverTime);
  if (parsed) app.serverOffsetMs = parsed.getTime() - Date.now();
}

function setProgress(id, value) {
  const element = $(id);
  if (!element) return;
  const progress = clamp(value);
  const bar = element.matches(".progress,.gauge") ? element.querySelector("i") : element;
  if (bar) bar.style.width = `${progress}%`;
  element.setAttribute("aria-valuenow", progress.toFixed(0));
}

function setStatus(id, status, label) {
  const element = $(id);
  if (!element) return;
  element.className = `status-chip ${status}`;
  const labelNode = element.querySelector("span");
  if (labelNode) labelNode.textContent = label;
}

function toast(title, message = "", type = "success") {
  const region = $("toast-region");
  if (!region) return;
  const item = node("div", `toast ${type === "success" ? "" : type}`.trim());
  item.dataset.toastId = String(++app.toastCounter);
  const content = node("div");
  content.append(node("strong", "", title));
  if (message) content.append(node("span", "", message));
  item.append(content);
  region.append(item);
  window.setTimeout(() => item.remove(), 4300);
}

async function api(path, options = {}) {
  const request = { ...options };
  request.headers = { ...(options.headers || {}) };
  if (request.body !== undefined) request.headers["Content-Type"] = "application/json";
  const response = await fetch(path, request);
  const payload = await response.json().catch(() => ({ message: response.statusText }));
  if (!response.ok || String(payload?.status || "").toLowerCase() === "error") {
    throw new Error(
      payload?.message || payload?.reason || payload?.error || response.statusText || "Request failed"
    );
  }
  return payload;
}

async function withPending(button, key, task) {
  if (!button || app.pending.has(key)) return;
  app.pending.add(key);
  button.disabled = true;
  button.classList.add("pending");
  button.setAttribute("aria-busy", "true");
  try {
    return await task();
  } finally {
    app.pending.delete(key);
    button.disabled = false;
    button.classList.remove("pending");
    button.removeAttribute("aria-busy");
  }
}

function updateConnectionStatus() {
  const elapsed = app.lastMessageAt ? Math.floor((Date.now() - app.lastMessageAt) / 1000) : null;
  const fullStateElapsed = app.lastFullStateAt
    ? Math.floor((Date.now() - app.lastFullStateAt) / 1000)
    : null;
  if (app.socket?.readyState === WebSocket.OPEN) {
    if (elapsed !== null && elapsed > 8) {
      setStatus("s-ws", "warning", `RECOVERING ${elapsed}s`);
      const staleSocket = app.socket;
      refreshFullState();
      staleSocket.close(4000, "stale stream");
    } else if (
      (fullStateElapsed !== null && fullStateElapsed > 5)
      || (fullStateElapsed === null && elapsed !== null && elapsed > 2)
    ) {
      // Fast packets intentionally contain quotes and system metrics only.
      // They must not make a stale positions/account snapshot look healthy.
      setStatus("s-ws", "warning", "SYNCING STATE");
      refreshFullState();
    } else if (elapsed !== null && elapsed > 4) {
      setStatus("s-ws", "warning", `STALE ${elapsed}s`);
    } else {
      setStatus("s-ws", "online", "STREAM LIVE");
    }
  }
}

function refreshFullState() {
  if (app.fallbackFetchPending) return;
  app.fallbackFetchPending = true;
  api("/api/state")
    .then((payload) => scheduleDashboardRender(payload, null))
    .catch(() => {})
    .finally(() => { app.fallbackFetchPending = false; });
}

function scheduleReconnect() {
  if (app.shuttingDown || app.reconnectTimer) return;
  app.reconnectAttempt += 1;
  const base = Math.min(30000, 800 * (2 ** Math.min(app.reconnectAttempt - 1, 6)));
  const delay = Math.round(base + Math.random() * Math.min(900, base * .22));
  let remaining = Math.ceil(delay / 1000);
  setStatus("s-ws", "offline", `RETRY ${remaining}s`);
  window.clearInterval(app.reconnectCountdown);
  app.reconnectCountdown = window.setInterval(() => {
    remaining -= 1;
    if (remaining > 0) setStatus("s-ws", "offline", `RETRY ${remaining}s`);
  }, 1000);
  app.reconnectTimer = window.setTimeout(() => {
    window.clearInterval(app.reconnectCountdown);
    app.reconnectTimer = null;
    connect();
  }, delay);
}

function connect() {
  if (app.shuttingDown || app.socket?.readyState === WebSocket.OPEN || app.socket?.readyState === WebSocket.CONNECTING) return;
  window.clearTimeout(app.reconnectTimer);
  app.reconnectTimer = null;
  setStatus("s-ws", "connecting", "CONNECTING");
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${protocol}://${location.host}/ws`);
  app.socket = socket;

  socket.addEventListener("open", () => {
    app.reconnectAttempt = 0;
    app.lastMessageAt = Date.now();
    setStatus("s-ws", "online", "STREAM LIVE");
  });

  socket.addEventListener("message", (event) => {
    let payload;
    try {
      payload = JSON.parse(event.data);
    } catch (error) {
      toast("Stream payload ignored", error.message, "warning");
      return;
    }
    app.lastMessageAt = Date.now();
    if (payload.ping) {
      updateConnectionStatus();
      return;
    }
    if (payload.server_time_utc) syncServerClock(payload.server_time_utc);
    if (payload._type === "live") {
      scheduleDashboardRender(null, payload);
      return;
    }
    const incomingSequence = finite(payload.sequence);
    if (incomingSequence && incomingSequence < app.lastSequence) return;
    if (incomingSequence) app.lastSequence = incomingSequence;
    scheduleDashboardRender(payload, null);
  });

  socket.addEventListener("close", () => {
    if (app.socket === socket) app.socket = null;
    setStatus("s-ws", "offline", "OFFLINE");
    scheduleReconnect();
  });

  socket.addEventListener("error", () => socket.close());
}

function scheduleDashboardRender(fullState = null, livePayload = null) {
  if (fullState) app.pendingFullState = fullState;
  if (livePayload) app.pendingLivePayload = livePayload;
  if (app.renderFrame !== null) return;
  app.renderFrame = window.requestAnimationFrame(() => {
    app.renderFrame = null;
    const nextFull = app.pendingFullState;
    const nextLive = app.pendingLivePayload;
    app.pendingFullState = null;
    app.pendingLivePayload = null;
    if (nextFull) {
      app.state = nextFull;
      app.lastFullStateAt = Date.now();
      captureTicks(nextFull.tick_stream || []);
      renderAll(nextFull);
    }
    if (nextLive) mergeLivePayload(nextLive);
  });
}

function mergeLivePayload(payload) {
  if (!app.state) app.state = {};
  if (payload.system) app.state.system = payload.system;
  if (payload.prices) app.state.prices = payload.prices;
  if (payload.tick_stream) app.state.tick_stream = payload.tick_stream;
  if (payload.sequence) {
    app.lastSequence = Math.max(app.lastSequence, finite(payload.sequence));
    app.state.sequence = app.lastSequence;
  }
  captureTicks(payload.tick_stream || []);
  renderSystem(app.state.system || {});
  renderWatch(app.state.prices || {}, app.state.tick_stream || [], app.state.market_fits || {});
  updateTemporalUi();
}

function captureTicks(ticks) {
  for (const tick of ticks) {
    if (!tick?.symbol) continue;
    const bid = Number(tick.bid);
    const ask = Number(tick.ask);
    if (!Number.isFinite(bid) || !Number.isFinite(ask)) continue;
    const key = `${tick.symbol}|${tick.time || tick.updated_at || ""}|${bid}|${ask}`;
    if (app.tickKeys.has(key)) continue;
    app.tickKeys.add(key);
    const values = app.tickHistory.get(tick.symbol) || [];
    values.push({ key, bid, ask, midpoint: (bid + ask) / 2 });
    if (values.length > 64) {
      const removed = values.shift();
      app.tickKeys.delete(removed.key);
    }
    app.tickHistory.set(tick.symbol, values);
  }
}

function renderAll(state) {
  renderHeader(state);
  renderMetrics(state);
  renderPositions(state.positions || [], state.account || {});
  renderWhenChanged(
    "pipeline",
    [
      state.symbol_decisions,
      state.market_fits,
      state.automation?.active_symbols,
      state.engine_running,
      (state.positions || []).map((position) => position.symbol),
    ],
    () => renderPipeline(state),
  );
  if (!app.watchInitialized) {
    renderWatch(state.prices || {}, state.tick_stream || [], state.market_fits || {});
    app.watchInitialized = true;
  }
  renderWhenChanged(
    "history",
    [state.closed_trades, state.account?.currency],
    () => renderHistory(state.closed_trades || [], state.account || {}),
  );
  renderWhenChanged("logs", state.logs, () => renderLogs(state.logs || []));
  renderWhenChanged(
    "readiness",
    [
      state.readiness?.ready,
      state.readiness?.severity,
      state.readiness?.code,
      state.readiness?.reason,
      state.readiness?.checks,
      state.automation?.safety_reason,
      state.account?.portfolio_risk_pct,
    ],
    () => renderReadiness(state),
  );
  renderWhenChanged(
    "decision",
    [state.llm, state.symbol_decisions, state.automation?.scan_status, state.automation?.last_scan],
    () => renderDecision(state),
  );
  renderWhenChanged(
    "capital-fits",
    [state.market_fits, state.account?.currency, state.shadow],
    () => {
      renderCapitalFits(state.market_fits || {}, state.account || {});
      renderShadowEvidence(state.shadow || {});
    },
  );
  renderWhenChanged(
    "evidence",
    {
      currency: state.account?.currency,
      total_trades: state.account?.total_trades,
      profit_factor: state.account?.profit_factor,
      expectancy_usd: state.account?.expectancy_usd,
      max_closed_drawdown_usd: state.account?.max_closed_drawdown_usd,
      strategy_evidence: state.account?.strategy_evidence,
    },
    () => renderEvidence(state.account || {}),
  );
  renderSystem(state.system || {});
  renderWhenChanged(
    "rules",
    [
      app.config,
      state.automation?.history_status,
      state.automation?.last_reconciled_at,
    ],
    () => renderRules(state),
  );
  updateTemporalUi();
}

function renderWhenChanged(key, value, renderer) {
  const fingerprint = JSON.stringify(value ?? null);
  if (app.renderHashes.get(key) === fingerprint) return false;
  app.renderHashes.set(key, fingerprint);
  renderer();
  return true;
}

function renderHeader(state) {
  const account = state.account || {};
  const automation = state.automation || {};
  const readiness = getReadiness(state);
  const brokerMode = String(account.account_mode || "UNKNOWN").toUpperCase();
  const executionMode = automation.dry_run ? "PAPER" : brokerMode;
  const modeChip = $("account-mode-chip");
  modeChip.textContent = executionMode;
  modeChip.className = `mode-chip ${executionMode.toLowerCase()}`;
  setText("account-identity-text", `${account.broker || "Pepperstone"} · ${account.server || "MT5"} · #${account.account_suffix || "—"}`);

  const terminalReady = Boolean(account.terminal_connected);
  const deterministic = String(automation.provider || app.config?.llm_provider || "").toLowerCase() === "deterministic";
  setStatus("s-account", terminalReady ? "online" : "offline", terminalReady ? `${brokerMode} MT5` : "MT5 OFFLINE");
  setStatus(
    "s-llm",
    automation.llm_online ? "online" : "warning",
    automation.llm_online ? (deterministic ? "RULES READY" : "LLM READY") : "DECISION CHECK",
  );
  if (automation.entries_armed) {
    setStatus("s-lock", readiness.ready ? "ready" : "warning", readiness.ready ? `ARMED · ${executionMode}` : "ARMED · GATED");
  } else {
    setStatus("s-lock", "neutral", "DISARMED");
  }

  const engineButton = $("engine-btn");
  engineButton.textContent = state.engine_running ? "Stop engine" : "Start engine";
  engineButton.className = `button ${state.engine_running ? "danger" : "secondary"}`;
  const armButton = $("arm-btn");
  armButton.textContent = automation.entries_armed ? "Disarm entries" : "Arm entries";
  armButton.className = `button ${automation.entries_armed ? "danger" : "primary"}`;
  const autonomyButton = $("autonomy-btn");
  const autonomyActive = Boolean(automation.autonomous_enabled);
  autonomyButton.textContent = autonomyActive
    ? `Autonomy ${String(automation.autonomous_status || "active").toLowerCase()}`
    : "Enable autonomy";
  autonomyButton.className = `button ${autonomyActive ? "danger" : "secondary"}`;
  autonomyButton.title = autonomyActive
    ? `Bound to account ending ${automation.autonomous_account_suffix || "—"} · last health check ${automation.last_health_check || "—"}`
    : "Enable persistent account-bound unattended operation";

  const modelConcurrency = Math.max(1, Math.floor(finite(app.config?.llm_max_concurrency, 1)));
  const candidates = Math.max(1, Math.floor(finite(app.config?.llm_entry_candidates_per_bar, 3)));
  setText("runtime-lane", `${modelConcurrency} concurrent · ${candidates} candidates / M5`);

  $("live-banner").hidden = executionMode !== "LIVE";
  setText("payload-sequence", state.sequence ? `Sequence ${state.sequence}` : "Sequence —");
}

function renderMetrics(state) {
  const account = state.account || {};
  const automation = state.automation || {};
  const currency = account.currency || "USD";
  const config = app.config || {};
  setText("m-equity", money(account.equity, currency));
  setText("m-balance", money(account.balance, currency));
  setText("m-floating", `Open P/L ${signedMoney(account.open_pl, currency)}`);
  setText("m-day", signedMoney(account.daily_pl, currency));
  colorValue($("m-floating"), account.open_pl);
  colorValue($("m-day"), account.daily_pl);
  const equityDelta = finite(account.equity) - finite(account.balance);
  setText("equity-delta", `${equityDelta >= 0 ? "+" : ""}${equityDelta.toFixed(2)}`);
  colorValue($("equity-delta"), equityDelta);
  setText("m-record", `${finite(account.total_trades)} closed · ${finite(account.win_rate_pct).toFixed(0)}% wins`);
  setText("m-risk", `${finite(account.portfolio_risk_pct).toFixed(2)}%`);
  setText("m-risk-usd", `${money(account.portfolio_risk_usd, currency)} at stops`);
  setText("m-margin", `${finite(account.margin_usage_pct).toFixed(1)}%`);
  setText("m-margin-level", `Margin level ${finite(account.margin_level_pct).toFixed(0)}%`);
  const brokerMode = String(account.account_mode || "UNKNOWN").toUpperCase();
  const executionMode = automation.dry_run ? "PAPER" : brokerMode;
  setText("m-mode", `${executionMode} EXEC · ${brokerMode} MT5 · 1:${finite(account.leverage)}`);

  const riskCap = Math.max(.01, finite(config.max_portfolio_risk_pct, 3));
  const marginCap = Math.max(.01, finite(config.max_margin_usage_pct, 35));
  setProgress("risk-progress", finite(account.portfolio_risk_pct) / riskCap * 100);
  setProgress("margin-progress", finite(account.margin_usage_pct) / marginCap * 100);
  setText("risk-limit-label", `cap ${riskCap.toFixed(1)}%`);
  setText("margin-limit-label", `cap ${marginCap.toFixed(0)}%`);
  const budget = finite(account.balance) * finite(config.risk_percent) / 100;
  setText("r-budget", money(budget, currency));
}

function appendCell(row, label, content, className = "") {
  const cell = node("td", className);
  cell.dataset.label = label;
  if (content instanceof Node) cell.append(content);
  else cell.textContent = String(content ?? "—");
  row.append(cell);
  return cell;
}

function stacked(primary, secondary, primaryClass = "") {
  const wrapper = node("div");
  wrapper.append(node("span", primaryClass, primary));
  if (secondary !== undefined && secondary !== "") wrapper.append(node("span", "subline", secondary));
  return wrapper;
}

function positionProgress(position) {
  const current = finite(position.current_price);
  const entry = finite(position.open_price);
  const stop = finite(position.sl);
  const target = finite(position.tp);
  if (!current || !entry || !stop || !target || stop === target) return null;
  const progress = position.direction === "SELL"
    ? (stop - current) / (stop - target) * 100
    : (current - stop) / (target - stop) * 100;
  return clamp(progress);
}

function renderPositions(positions, account) {
  const body = $("positions");
  const currency = account.currency || "USD";
  setText("position-count", `${positions.length} ACTIVE`);
  body.replaceChildren();
  if (!positions.length) {
    const row = node("tr");
    const cell = node("td", "empty-state", "No open positions");
    cell.colSpan = 9;
    row.append(cell);
    body.append(row);
    return;
  }

  const fragment = document.createDocumentFragment();
  for (const position of positions) {
    const row = node("tr");
    const market = stacked(position.symbol || "—", position.bot_owned ? `BOT · ${position.magic || "—"}` : "EXTERNAL", position.bot_owned ? "ownership" : "muted");
    appendCell(row, "Market", market);
    const side = node("span", `side-badge ${String(position.direction).toLowerCase()}`, position.direction || "—");
    appendCell(row, "Side", side);
    appendCell(row, "Size", finite(position.lot).toFixed(2), "mono");
    appendCell(row, "Open / current", stacked(price(position.open_price), price(position.current_price), "mono"), "mono");
    const protection = stacked(`SL ${price(position.sl)}`, `TP ${price(position.tp)}`, "negative");
    appendCell(row, "SL / TP", protection, "mono");

    const pl = stacked(signedMoney(position.profit_usd, currency), `${finite(position.profit_pips).toFixed(1)} pips`, finite(position.profit_usd) >= 0 ? "positive" : "negative");
    const journey = positionProgress(position);
    if (journey !== null) {
      const progress = node("div", "progress row-progress");
      const bar = node("i");
      bar.style.width = `${journey}%`;
      progress.append(bar);
      pl.append(progress);
    }
    appendCell(row, "P/L", pl, "mono");
    appendCell(row, "Stop risk", stacked(money(position.risk_to_sl_usd, currency), `${finite(position.risk_pct_balance).toFixed(2)}%`, finite(position.risk_pct_balance) > 2 ? "negative" : ""), "mono");
    const liveR = finite(position.risk_to_sl_usd) > 0 ? finite(position.profit_usd) / finite(position.risk_to_sl_usd) : 0;
    const management = stacked(
      `${finite(position.planned_rr).toFixed(2)} R plan`,
      `${liveR >= 0 ? "+" : ""}${liveR.toFixed(2)} R live · peak ${money(position.peak_profit_usd, currency)}`,
      position.profit_lock_armed ? "positive" : ""
    );
    if (position.profit_lock_armed) management.append(node("small", "ownership", `BROKER FLOOR ${money(position.profit_lock_floor_usd, currency)} EST.`));
    if (finite(position.profit_retention_floor_usd) > 0) management.append(node("small", "ownership", `NET RETENTION ${money(position.profit_retention_floor_usd, currency)}`));
    appendCell(row, "Plan", management, "mono");
    const manage = node("button", "button secondary", "Manage");
    manage.type = "button";
    manage.dataset.manageTicket = String(position.ticket);
    appendCell(row, "Actions", manage);
    fragment.append(row);
  }
  body.append(fragment);
}

function stageIndex(stage) {
  const value = String(stage || "WAITING").toUpperCase();
  if (value.includes("ORDER") || value.includes("OPENED") || value.includes("SUBMITTED") || value.includes("EXECUTION")) return 4;
  if (value.includes("RISK") || value.includes("PLAN") || value.includes("SIGNAL") || value.includes("REJECTED") || value.includes("CONFIDENCE")) return 3;
  if (value.includes("LLM") || value.includes("DECIDED")) return 2;
  if (value.includes("ANALYZ") || value.includes("DATA")) return 1;
  return 0;
}

function stageClass(stage) {
  const value = String(stage || "WAITING").toUpperCase();
  if (value.includes("OPENED") || value.includes("SUBMITTED")) return "opened";
  if (value.includes("ERROR")) return "error";
  if (value.includes("REJECT") || value.includes("BLOCK") || value.includes("BELOW")) return "rejected";
  if (value.includes("ANALYZ") || value.includes("LLM") || value.includes("RISK") || value.includes("ORDER")) return "analyzing";
  return "waiting";
}

function decisionSymbols(state) {
  const active = state.automation?.active_symbols;
  if (Array.isArray(active)) {
    const scoped = new Set(active.filter(Boolean));
    for (const symbol of Object.keys(state.market_fits || {})) scoped.add(symbol);
    for (const position of state.positions || []) {
      if (position?.symbol) scoped.add(position.symbol);
    }
    if (scoped.size || state.engine_running) return [...scoped];
  }
  const values = new Set();
  for (const symbol of app.config?.symbols || []) values.add(symbol);
  for (const source of [state.prices, state.symbol_decisions, state.market_fits]) {
    for (const symbol of Object.keys(source || {})) values.add(symbol);
  }
  return [...values];
}

function renderPipeline(state) {
  const container = $("symbol-pipeline");
  const decisions = state.symbol_decisions || {};
  const symbols = decisionSymbols(state);
  container.replaceChildren();
  if (!symbols.length) {
    container.append(node("div", "empty-card", "Waiting for configured symbols."));
    setText("pipeline-summary", "No symbol state available");
    return;
  }

  const threshold = normalizeConfidence(app.config?.confidence_threshold ?? .7);
  const deterministic = String(state.automation?.provider || app.config?.llm_provider || "").toLowerCase() === "deterministic";
  const activeSymbols = new Set(state.automation?.active_symbols || []);
  const positionSymbols = new Set((state.positions || []).map((position) => position.symbol));
  const fits = state.market_fits || {};
  const active = [];
  const fragment = document.createDocumentFragment();
  for (const symbol of symbols) {
    const fit = fits[symbol] || {};
    const standby = Boolean(
      Object.keys(fit).length
      && !activeSymbols.has(symbol)
      && !positionSymbols.has(symbol)
    );
    const decision = standby
      ? {
          symbol,
          stage: String(fit.status || "").toUpperCase() === "MARKET CLOSED" ? "MARKET CLOSED" : "STANDBY",
          action: "HOLD",
          confidence: 0,
          gate_reason: fit.reason || "Not selected for active decision processing.",
          updated_at: decisions[symbol]?.updated_at || "",
        }
      : (decisions[symbol] || { symbol, stage: "WAITING", action: "—", confidence: 0 });
    const stage = String(decision.stage || "WAITING");
    const statusClass = stageClass(stage);
    if (statusClass === "analyzing") active.push(symbol);
    const card = node("article", `pipeline-card ${statusClass}`);
    card.dataset.symbol = symbol;
    card.setAttribute("aria-label", `${symbol} decision stage ${stage}`);

    const top = node("div", "pipeline-top");
    top.append(node("strong", "pipeline-symbol", symbol), node("span", "stage-badge", stage));
    card.append(top);

    const track = node("div", "stage-track");
    track.setAttribute("aria-hidden", "true");
    const index = stageIndex(stage);
    for (let step = 0; step < 5; step += 1) {
      track.append(node("i", step < index ? "done" : step === index ? "current" : ""));
    }
    card.append(track);

    const action = String(decision.action || "—").toUpperCase();
    const confidence = normalizeConfidence(decision.confidence);
    const decisionRow = node("div", "pipeline-decision");
    decisionRow.append(node("strong", `pipeline-action ${action.toLowerCase()}`, action), node("span", "pipeline-confidence", `${confidence.toFixed(0)}% ${deterministic ? "rules" : "model"}`));
    card.append(decisionRow);
    const mini = node("div", "mini-confidence");
    const fill = node("i");
    fill.style.width = `${confidence}%`;
    const marker = node("b");
    marker.style.left = `${threshold}%`;
    mini.append(fill, marker);
    card.append(mini);
    card.append(node("div", "pipeline-reason", decision.gate_reason || decision.reasoning || "Waiting for a completed M5 candle."));
    const meta = node("div", "pipeline-meta");
    meta.append(node("span", "", decision.inference_time_s ? `${finite(decision.inference_time_s).toFixed(2)}s ${deterministic ? "evaluation" : "inference"}` : "not evaluated"));
    const decisionTime = node("time", "", decision.updated_at ? ageLabel(decision.updated_at) : "not scanned");
    if (decision.updated_at) decisionTime.dataset.timestamp = decision.updated_at;
    meta.append(decisionTime);
    card.append(meta);
    if (decision.manual_override_available) {
      const override = node("div", "pipeline-override");
      const review = node("button", "button secondary", "Review rejected trade");
      review.type = "button";
      review.dataset.reviewRejected = symbol;
      override.append(review);
      card.append(override);
    }
    fragment.append(card);
  }
  container.append(fragment);
  const selectedCount = symbols.filter((symbol) => activeSymbols.has(symbol)).length;
  setText(
    "pipeline-summary",
    active.length
      ? `Analyzing ${active.join(", ")}`
      : `${symbols.length} candidates · ${selectedCount} active · one decision per completed M5 candle`,
  );
}

function createQuoteCard(symbol) {
  const card = node("article", "quote-card");
  card.dataset.symbol = symbol;
  const top = node("div", "quote-top");
  top.append(node("strong", "quote-symbol", symbol), node("span", "quote-asset", "—"));
  const meta = node("div", "quote-meta");
  meta.append(node("span", "quote-trend", "Trend —"), node("span", "quote-adx", "ADX —"));
  const prices = node("div", "quote-prices");
  const sell = node("div", "quote-side sell-side");
  sell.append(node("div", "side-label", "SELL"), node("div", "side-price sell-price", "—"));
  const buy = node("div", "quote-side buy-side");
  buy.append(node("div", "side-label", "BUY"), node("div", "side-price buy-price", "—"));
  prices.append(sell, buy);
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.classList.add("sparkline");
  svg.setAttribute("viewBox", "0 0 160 40");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", `${symbol} recent midpoint ticks`);
  const grid = document.createElementNS("http://www.w3.org/2000/svg", "line");
  grid.classList.add("gridline");
  grid.setAttribute("x1", "0"); grid.setAttribute("x2", "160"); grid.setAttribute("y1", "20"); grid.setAttribute("y2", "20");
  const line = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
  svg.append(grid, line);
  const footer = node("div", "quote-footer");
  footer.append(node("span", "spread-health", "spread —"), node("time", "freshness", "price —"));
  card.append(top, meta, prices, svg, footer);
  return card;
}

function flashValue(element, next) {
  const value = Number(next);
  const previous = Number(element.dataset.value);
  element.textContent = price(value);
  element.dataset.value = String(value);
  if (Number.isFinite(previous) && Number.isFinite(value) && previous !== value) {
    const now = performance.now();
    const lastFlash = finite(element.dataset.flashAt);
    if (now - lastFlash < 700) return;
    element.dataset.flashAt = String(now);
    element.classList.remove("tick-up", "tick-down");
    window.requestAnimationFrame(() => {
      element.classList.add(value > previous ? "tick-up" : "tick-down");
    });
  }
}

function renderSparkline(svg, symbol) {
  const history = app.tickHistory.get(symbol) || [];
  const line = svg.querySelector("polyline");
  if (history.length < 2) {
    line.setAttribute("points", "");
    svg.classList.remove("down");
    return;
  }
  const values = history.map((item) => item.midpoint);
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  const range = maximum - minimum || 1;
  const points = values.map((value, index) => {
    const x = index / (values.length - 1) * 160;
    const y = 35 - ((value - minimum) / range * 30);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  line.setAttribute("points", points);
  svg.classList.toggle("down", values.at(-1) < values[0]);
}

function renderWatch(prices, ticks, fits = {}) {
  captureTicks(ticks);
  const container = $("watch");
  const symbols = [...new Set([
    ...Object.keys(fits || {}),
    ...Object.keys(prices || {}),
  ])];
  const maxTickAge = Math.max(1, finite(app.config?.max_tick_age_seconds, 10));
  const freshCount = symbols.filter((symbol) => {
    const updated = prices?.[symbol]?.updated_at;
    return updated && finite(ageSeconds(updated), 999999) <= maxTickAge;
  }).length;
  setText(
    "watch-count",
    symbols.length ? `${symbols.length} BROKER MARKETS · ${freshCount} FRESH` : "WAITING FOR PRICES",
  );
  if (!symbols.length) {
    container.replaceChildren(node("div", "empty-card", "Waiting for live quotes."));
    return;
  }
  container.querySelector(".empty-card")?.remove();
  for (const [symbol, existing] of app.quoteCards) {
    if (!symbols.includes(symbol)) {
      existing.remove();
      app.quoteCards.delete(symbol);
    }
  }

  const maxSpread = Math.max(.01, finite(app.config?.max_spread_pips, 3));
  for (const symbol of symbols) {
    const quote = prices[symbol] || {};
    const fit = fits[symbol] || {};
    let card = app.quoteCards.get(symbol);
    if (!card) {
      card = createQuoteCard(symbol);
      app.quoteCards.set(symbol, card);
      container.append(card);
    }
    card.querySelector(".quote-symbol").textContent = symbol;
    card.querySelector(".quote-asset").textContent = quote.asset_class || fit.asset_class || "FX/CFD";
    card.querySelector(".quote-trend").textContent = `Trend ${quote.trend || fit.selection_regime || "—"}`;
    card.querySelector(".quote-adx").textContent = `ADX ${finite(quote.adx, finite(fit.selection_adx)).toFixed(0)}`;

    // Intentional user-requested presentation only. Broker execution remains BUY at ASK / SELL at BID.
    flashValue(card.querySelector(".sell-price"), quote.ask);
    flashValue(card.querySelector(".buy-price"), quote.bid);

    const spread = finite(
      quote.spread_value,
      finite(quote.spread_pips, finite(fit.spread_value)),
    );
    const unit = quote.spread_unit || fit.spread_unit || "pips";
    const spreadNode = card.querySelector(".spread-health");
    spreadNode.textContent = `spread ${spread.toFixed(1)} ${unit}`;
    const ratio = spread / maxSpread;
    spreadNode.className = `spread-health ${ratio > 1 ? "bad" : ratio > .7 ? "warn" : ""}`.trim();
    const freshness = card.querySelector(".freshness");
    freshness.textContent = quote.updated_at
      ? ageLabel(quote.updated_at)
      : (fit.status || "WAITING FOR BROKER TICK");
    freshness.classList.toggle(
      "stale",
      quote.updated_at
        ? finite(ageSeconds(quote.updated_at), 999999) > maxTickAge
        : true,
    );
    renderSparkline(card.querySelector(".sparkline"), symbol);
  }
}

function renderHistory(trades, account) {
  const body = $("history");
  const currency = account.currency || "USD";
  setText("history-count", `${trades.length} RECENT`);
  body.replaceChildren();
  if (!trades.length) {
    const row = node("tr");
    const cell = node("td", "empty-state", "No reconciled bot trades yet");
    cell.colSpan = 8;
    row.append(cell);
    body.append(row);
    return;
  }
  const fragment = document.createDocumentFragment();
  for (const trade of trades) {
    const row = node("tr");
    appendCell(row, "Closed", String(trade.close_time || "—").replace("T", " ").slice(0, 19), "mono");
    appendCell(row, "Market", node("strong", "", trade.symbol || "—"));
    appendCell(row, "Side", node("span", `side-badge ${String(trade.direction).toLowerCase()}`, trade.direction || "—"));
    appendCell(row, "Size", finite(trade.lot).toFixed(2), "mono");
    appendCell(row, "Open / close", `${price(trade.open_price)} / ${price(trade.close_price)}`, "mono");
    const result = stacked(
      signedMoney(trade.profit_usd, currency),
      `${finite(trade.profit_pips).toFixed(1)} pips · ${finite(trade.rr_achieved) >= 0 ? "+" : ""}${finite(trade.rr_achieved).toFixed(2)} R`,
      finite(trade.profit_usd) >= 0 ? "positive" : "negative",
    );
    appendCell(row, "Net P/L", result, "mono");
    appendCell(
      row,
      "MFE / MAE",
      stacked(
        `MFE ${signedMoney(trade.mfe_usd, currency)} · ${finite(trade.mfe_pips).toFixed(1)}p`,
        `MAE ${signedMoney(trade.mae_usd, currency)} · ${finite(trade.mae_pips).toFixed(1)}p`,
        "mono",
      ),
      "mono",
    );
    appendCell(row, "Exit", String(trade.close_reason || "BROKER"), "mono");
    fragment.append(row);
  }
  body.append(fragment);
}

function renderLogs(logs) {
  const container = $("logs-tab");
  const nearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 36;
  container.replaceChildren();
  const fragment = document.createDocumentFragment();
  for (const entry of logs.slice(-150)) {
    const text = String(entry);
    const className = text.includes("[ERROR]") ? "log-line error" : text.includes("[WARN") ? "log-line warning" : "log-line";
    fragment.append(node("div", className, text));
  }
  if (!logs.length) fragment.append(node("div", "empty-card", "No system log entries."));
  container.append(fragment);
  if (nearBottom) container.scrollTop = container.scrollHeight;
}

function getReadiness(state) {
  if (state.readiness && typeof state.readiness.ready === "boolean") return state.readiness;
  const account = state.account || {};
  const automation = state.automation || {};
  const checks = [
    { code: "ENGINE", label: "Decision engine", ok: Boolean(state.engine_running), detail: state.engine_running ? "Running" : "Stopped" },
    { code: "CONNECTION", label: "Pepperstone connection", ok: Boolean(account.terminal_connected), detail: account.terminal_connected ? "Connected" : "Disconnected" },
    { code: "LLM", label: "Decision provider", ok: Boolean(automation.llm_online), detail: automation.llm_online ? "Available" : "Unavailable" },
    { code: "ARMED", label: "Entry authorization", ok: Boolean(automation.entries_armed), detail: automation.entries_armed ? "Armed" : "Disarmed" },
  ];
  const failed = checks.find((check) => !check.ok);
  return { ready: !failed, severity: failed ? "BLOCKED" : "READY", code: failed?.code || "READY", reason: failed?.detail || "Broker channel ready", checks };
}

function renderReadiness(state) {
  const readiness = getReadiness(state);
  const automation = state.automation || {};
  const panel = $("readiness");
  panel.classList.toggle("ready", Boolean(readiness.ready));
  panel.classList.toggle("blocked", !readiness.ready);
  const badge = $("safety-state");
  badge.textContent = readiness.ready ? "READY" : String(readiness.severity || "BLOCKED");
  badge.className = `state-badge ${readiness.ready ? "ready" : String(readiness.severity).toUpperCase() === "WARNING" ? "warning" : "blocked"}`;
  const blockedLabels = {
    ARMED: "ENTRY LOCKED",
    ENGINE: "ENGINE STOPPED",
    LOOP: "ENGINE STALLED",
    CONNECTION: "MT5 OFFLINE",
    ACCOUNT: "ACCOUNT BLOCK",
    HISTORY: "HISTORY BLOCK",
    PERMISSIONS: "PERMISSION BLOCK",
    LLM: "DECISION BLOCK",
    POSITIONS: "POSITION DATA BLOCK",
    PORTFOLIO: "RISK BLOCK",
    STOP_RISK: "RISK UNKNOWN",
    DAILY_LOSS: "DAILY STOP",
  };
  const readinessCode = String(readiness.code || "BLOCKED").toUpperCase();
  setText("ready-state", readiness.ready ? "BROKER READY" : blockedLabels[readinessCode] || readinessCode);
  setText("ready-reason", readiness.reason || "Execution checks are incomplete.");
  setText("safety-reason", automation.safety_reason || readiness.reason || "New entries are disarmed.");

  const list = $("readiness-checks");
  list.replaceChildren();
  for (const check of readiness.checks || []) {
    const item = node("li", check.ok ? "passed" : "failed");
    item.append(node("i", "", check.ok ? "✓" : "×"), node("span", "", check.label || check.code || "Check"), node("small", "", check.detail || (check.ok ? "Ready" : "Blocked")));
    list.append(item);
  }
  if (!(readiness.checks || []).length) {
    const item = node("li", "pending");
    item.append(node("i", "", "·"), node("span", "", "Waiting for readiness checks"));
    list.append(item);
  }

  const account = state.account || {};
  const riskCap = finite(app.config?.max_portfolio_risk_pct, 3);
  const overRisk = finite(account.portfolio_risk_pct) > riskCap;
  const criticalCodes = new Set(["LOOP", "PORTFOLIO", "STOP_RISK", "POSITIONS", "DAILY_LOSS", "DRAWDOWN", "ACCOUNT", "CONNECTION", "PERMISSIONS", "HISTORY"]);
  const showAlert = overRisk || (!readiness.ready && criticalCodes.has(String(readiness.code || "").toUpperCase()));
  const banner = $("risk-banner");
  banner.hidden = !showAlert;
  if (showAlert) {
    setText("risk-banner-title", overRisk ? "Portfolio risk lock" : "Execution channel blocked");
    setText("risk-banner-text", readiness.reason || `Stop risk exceeds the ${riskCap.toFixed(2)}% cap.`);
  }
}

function latestSymbolDecision(state) {
  const decisions = Object.values(state.symbol_decisions || {});
  if (!decisions.length) return null;
  return [...decisions].sort((left, right) => {
    const leftTime = parseDate(left.updated_at)?.getTime() || 0;
    const rightTime = parseDate(right.updated_at)?.getTime() || 0;
    return rightTime - leftTime;
  })[0];
}

function renderDecision(state) {
  const automation = state.automation || {};
  const globalDecision = state.llm || {};
  const recent = latestSymbolDecision(state);
  const decision = recent || {
    symbol: globalDecision.last_symbol,
    action: globalDecision.last_action,
    confidence: globalDecision.confidence,
    reasoning: globalDecision.reasoning,
    inference_time_s: globalDecision.inference_time_s,
    updated_at: globalDecision.last_updated,
  };
  const action = String(decision.action || "—").toUpperCase();
  const confidence = normalizeConfidence(decision.confidence);
  const threshold = normalizeConfidence(app.config?.confidence_threshold ?? .7);
  const provider = globalDecision.provider || automation.provider || app.config?.llm_provider || "local";
  const deterministic = String(provider).toLowerCase() === "deterministic";
  setText("decision-symbol", decision.symbol || "—");
  const actionNode = $("decision-action");
  actionNode.textContent = action;
  actionNode.className = `decision-action ${action.toLowerCase()}`;
  setText("decision-confidence", `${confidence.toFixed(0)}% ${deterministic ? "rules" : "model"}`);
  $("confidence-bar").style.width = `${confidence}%`;
  $("confidence-track").setAttribute("aria-valuenow", confidence.toFixed(0));
  $("confidence-threshold-marker").style.left = `${threshold}%`;
  $("confidence-threshold-marker").querySelector("span").textContent = `Gate ${threshold.toFixed(0)}%`;
  setText("decision-latency", decision.inference_time_s ? `${finite(decision.inference_time_s).toFixed(2)}s` : "—");
  const model = globalDecision.model || automation.model || app.config?.decision_model || "—";
  setText("decision-model", `${String(provider).toUpperCase()} · ${model}`);
  setText(
    "decision-role",
    deterministic
      ? "Deterministic direction and exits"
      : "Entry confirm / veto; exits deterministic",
  );
  setText("scan-timeframe", automation.scan_timeframe || "M5 close");
  setText("scan-confirmation", automation.confirmation_timeframes || "M15 / H1 / H4");
  const scanStatus = automation.scan_status || decision.stage || "NOT STARTED";
  setText("scan-status", deterministic ? String(scanStatus).replaceAll("LLM", "RULES") : scanStatus);
  setText("last-scan", automation.last_scan || (decision.updated_at ? formatTimestamp(decision.updated_at) : "—"));
  setText("decision-reason", decision.gate_reason || decision.reasoning || "Waiting for a completed candle.");
  setText("llm-time", decision.updated_at ? formatTimestamp(decision.updated_at) : globalDecision.last_updated || "—");
}

function renderCapitalFits(fits, account) {
  const container = $("capital-fit-grid");
  const values = Object.entries(fits || {}).sort((left, right) => {
    const a = left[1] || {};
    const b = right[1] || {};
    if (Boolean(a.selected) !== Boolean(b.selected)) return a.selected ? -1 : 1;
    return finite(b.selection_score) - finite(a.selection_score);
  });
  const currency = account.currency || "USD";
  container.replaceChildren();
  if (!values.length) {
    container.append(node("div", "empty-card", "Capital diagnostics appear after M5 analysis."));
    setText("market-fit-count", "ASSESSING");
    setText("capital-fit-summary", "Assessing");
    setText("budget-detail", "Execution ceiling only; excluded from market ranking");
    return;
  }
  const fitCount = values.filter(([, fit]) => fit.capital_fit).length;
  setText("market-fit-count", `${fitCount}/${values.length} FIT`);
  const selectedCount = values.filter(([, fit]) => fit.selected).length;
  const readyCount = values.filter(([, fit]) => fit.model_eligible).length;
  setText("capital-fit-summary", `${values.length} assessed / ${selectedCount} active / ${readyCount} ready for review`);
  const lowestRisk = Math.min(...values.map(([, fit]) => finite(fit.min_stop_risk_usd, Infinity)));
  if (Number.isFinite(lowestRisk)) setText("budget-detail", `Execution only: lowest current stop risk ${money(lowestRisk, currency)}`);

  const primary = document.createDocumentFragment();
  const secondary = document.createDocumentFragment();
  for (const [index, [symbol, fit]] of values.entries()) {
    const setupBlocked = String(fit.status || "").toUpperCase() === "SETUP BLOCK";
    const card = node("article", `capital-fit-card ${fit.capital_fit ? "fit" : "blocked"}`);
    const top = node("div", "fit-top");
    const statusText = !fit.broker_open
      ? "MARKET CLOSED"
      : setupBlocked
        ? "SETUP BLOCK · MARKET OPEN"
        : fit.status || (fit.capital_fit ? "CAPITAL FIT" : "BLOCKED");
    top.append(node("strong", "", symbol), node("span", "fit-status", statusText));
    card.append(top, node("p", "fit-reason", fit.reason || "Awaiting broker feasibility details."));
    const opportunityReasons = [
      fit.entry_prefilter_reason,
      ...Object.entries(fit.opportunity_rejections || {}).map(([side, reason]) => `${side}: ${reason}`),
    ].filter(Boolean);
    if (opportunityReasons.length) card.append(node("p", "fit-reason", opportunityReasons.join("; ")));
    const stats = node("div", "fit-stats");
    const statValues = [
      ["Min stop risk", finite(fit.min_stop_risk_usd) > 0 ? money(fit.min_stop_risk_usd, currency) : "—"],
      ["Min margin", finite(fit.min_margin_usd) > 0 ? money(fit.min_margin_usd, currency) : "—"],
      ["Projected use", finite(fit.projected_margin_pct) > 0 ? `${finite(fit.projected_margin_pct).toFixed(1)}%` : "—"],
      ["Spread", `${finite(fit.spread_value).toFixed(1)} ${fit.spread_unit || ""}`],
      ["Adaptive score", `${finite(fit.selection_score).toFixed(1)}/100`],
      ["Opportunity", fit.opportunity_status || "ASSESSING"],
      ["Verified directions", (fit.viable_entry_actions || []).join(" / ") || "NONE"],
      ["Selection", fit.model_selected
        ? `MODEL QUEUE · #${fit.model_selection_rank || "—"}`
        : fit.selected
          ? `DETERMINISTIC · #${fit.selection_rank || "—"}`
          : "STANDBY"],
    ];
    for (const [label, value] of statValues) {
      const stat = node("div");
      stat.append(node("span", "", label), node("b", "", value));
      stats.append(stat);
    }
    card.append(stats);
    card.append(node(
      "p",
      "fit-footnote",
      `Minimum ${finite(fit.min_volume).toFixed(2)} lot · ${finite(fit.min_margin_usd) > 0 ? money(fit.min_margin_usd, currency) : "margin pending"}`,
    ));
    const directions = node("div", "direction-fit");
    for (const side of ["BUY", "SELL"]) {
      const sideFit = fit.directions?.[side]?.capital_fit;
      const sideLabel = sideFit
        ? `${side} FITS`
        : !fit.broker_open
          ? `${side} MARKET CLOSED`
          : setupBlocked
            ? `${side} SETUP BLOCKED`
            : `${side} BLOCKED`;
      directions.append(node("span", sideFit ? "ok" : "", sideLabel));
    }
    card.append(directions);
    (index < 4 ? primary : secondary).append(card);
  }
  container.append(primary);
  if (values.length > 4) {
    const more = node("details", "capital-fit-more");
    const summary = node("summary", "", `Show ${values.length - 4} more markets`);
    const grid = node("div", "capital-fit-more-grid");
    grid.append(secondary);
    more.append(summary, grid);
    container.append(more);
  }
}

function renderShadowEvidence(shadow) {
  const exitResolved = Math.max(0, Math.floor(finite(shadow.exit_resolved)));
  const exitPending = Math.max(0, Math.floor(finite(shadow.exit_pending)));
  const exitHorizons = Array.isArray(shadow.exit_horizon_breakdown)
    ? shadow.exit_horizon_breakdown
    : [];
  setText(
    "exit-counterfactual",
    !shadow.exit_enabled
      ? "Post-exit hold comparison is disabled."
      : exitResolved
        ? `Post-exit original-bracket replay: ${exitResolved} resolved, ${exitPending} pending · hold minus actual ${finite(shadow.exit_average_delta_r).toFixed(2)} R average (${Math.floor(finite(shadow.exit_held_better))} hold-better, ${Math.floor(finite(shadow.exit_actual_better))} actual-better). ${exitHorizons.map((item) => `${Math.floor(finite(item.minutes))}m ${finite(item.average_delta_r).toFixed(2)} R`).join(" · ")}. Diagnostic only.`
        : `${exitPending} post-exit hold comparison${exitPending === 1 ? "" : "s"} pending; live exit rules are never changed automatically.`,
  );
  if (!shadow.enabled) {
    setText("shadow-summary", "Rejected-signal outcome evaluation is disabled.");
    setText("shadow-gates", "Recent gate outcomes are unavailable.");
    setText("shadow-directions", "BUY/SELL execution balance is unavailable.");
    return;
  }
  const resolved = Math.max(0, Math.floor(finite(shadow.resolved)));
  const pending = Math.max(0, Math.floor(finite(shadow.pending)));
  const windowHours = Math.max(1, Math.floor(finite(shadow.evidence_window_hours) || 48));
  if (!resolved) {
    setText("shadow-summary", `${pending} rejected signal${pending === 1 ? "" : "s"} awaiting broker-price outcomes; no rule changes are made from this data yet.`);
    setText("shadow-gates", `No rejected signals have resolved in the last ${windowHours} hours.`);
    setText("shadow-directions", "No BUY/SELL candidate evidence is available for the rolling window.");
    return;
  }
  setText(
    "shadow-summary",
    `Rejected-signal evidence: ${resolved} resolved, ${pending} pending · ${finite(shadow.win_rate_pct).toFixed(1)}% positive · ${finite(shadow.expectancy_r).toFixed(2)} R average. Diagnostic only.`,
  );
  const gates = Array.isArray(shadow.gate_breakdown) ? shadow.gate_breakdown : [];
  setText(
    "shadow-gates",
    gates.length
      ? `Last ${windowHours}h gate outcomes: ${gates.map((gate) => `${gate.gate} ${gate.wins}/${gate.resolved} positive, ${finite(gate.expectancy_r).toFixed(2)} R`).join(" · ")}.`
      : `No rejected signals have resolved in the last ${windowHours} hours.`,
  );
  const directions = Array.isArray(shadow.direction_breakdown)
    ? shadow.direction_breakdown
    : [];
  setText(
    "shadow-directions",
    directions.length
      ? `${windowHours}h direction funnel: ${directions.map((side) => {
          const action = String(side.action || "").toUpperCase();
          const candidates = Math.max(0, Math.floor(finite(side.candidates)));
          const executed = Math.max(0, Math.floor(finite(side.executed)));
          const rejected = Math.max(0, Math.floor(finite(side.rejected)));
          const shadowResolved = Math.max(0, Math.floor(finite(side.shadow_resolved)));
          return `${action} ${executed}/${candidates} executed, ${rejected} rejected, `
            + `${shadowResolved} replayed at ${finite(side.shadow_expectancy_r).toFixed(2)} R`;
        }).join(" · ")}. Diagnostic only; it never changes entry rules automatically.`
      : "No BUY/SELL candidate evidence is available for the rolling window.",
  );
}

function renderEvidence(account) {
  const currency = account.currency || "USD";
  const sample = Math.max(0, Math.floor(finite(account.total_trades)));
  const enoughSample = sample >= 30;
  const evidence = String(account.strategy_evidence || "").toUpperCase();
  const positiveInSample = enoughSample && evidence.includes("POSITIVE");
  const negativeEvidence = enoughSample && evidence.includes("NEGATIVE");
  const profitFactor = Number(account.profit_factor);
  setText("evidence-profit-factor", Number.isFinite(profitFactor) ? profitFactor.toFixed(2) : "—");
  setText("evidence-expectancy", money(account.expectancy_usd, currency));
  setText("evidence-drawdown", money(account.max_closed_drawdown_usd, currency));
  setText("evidence-sample", `${sample} trade${sample === 1 ? "" : "s"}`);
  const status = $("evidence-status");
  status.textContent = !enoughSample ? "UNVALIDATED" : positiveInSample ? "POSITIVE IN-SAMPLE" : negativeEvidence ? "NEGATIVE" : "INCONCLUSIVE";
  status.className = `state-badge ${negativeEvidence ? "blocked" : "warning"}`;
  const note = $("evidence-note");
  note.classList.toggle("validated", positiveInSample);
  const noteBadge = note.querySelector(".state-badge");
  noteBadge.textContent = positiveInSample ? "IN-SAMPLE ONLY" : !enoughSample ? "UNVALIDATED" : "EVIDENCE";
  const message = note.querySelector("span:last-child");
  message.textContent = enoughSample
    ? `${sample} closed trades: ${evidence || "inconclusive evidence"}. Walk-forward validation is still required; this is not a profit forecast.`
    : `Only ${sample} of 30 minimum closed trades. These statistics are descriptive evidence, not a profit forecast.`;
}

function renderSystem(system) {
  const cpu = clamp(system.cpu_pct);
  const gpu = clamp(system.gpu_util_pct);
  const vram = clamp(system.gpu_mem_pct);
  const ram = clamp(system.ram_pct);
  const usedVram = finite(system.gpu_mem_used_mb) / 1024;
  const totalVram = finite(system.gpu_mem_total_mb) / 1024;
  const temperature = finite(system.gpu_temp_c);
  setText("sys-cpu", `${cpu.toFixed(0)}%`);
  setText("sys-gpu", `${gpu.toFixed(0)}%${temperature ? ` · ${temperature.toFixed(0)}°C` : ""}`);
  setText("sys-vram", totalVram ? `${usedVram.toFixed(1)} / ${totalVram.toFixed(1)} GB · ${vram.toFixed(0)}%` : `${vram.toFixed(0)}%`);
  setText("sys-ram", `${finite(system.ram_used_gb).toFixed(1)} / ${finite(system.ram_total_gb).toFixed(0)} GB · ${ram.toFixed(0)}%`);
  setText("diagnostics-summary", `GPU ${gpu.toFixed(0)}% · VRAM ${vram.toFixed(0)}%`);
  setProgress("cpu-progress", cpu);
  setProgress("gpu-progress", gpu);
  setProgress("vram-progress", vram);
  setProgress("ram-progress", ram);
}

function renderRules(state) {
  const config = app.config;
  if (!config) return;
  setText("r-trade", `${finite(config.risk_percent).toFixed(2)}%`);
  setText("r-portfolio", `${finite(config.max_portfolio_risk_pct).toFixed(2)}%`);
  setText("r-rr", `${finite(config.min_risk_reward_ratio).toFixed(2)}:1`);
  setText("r-margin", `${finite(config.max_margin_usage_pct).toFixed(1)}%`);
  setText("r-sync", state.automation?.history_status || state.automation?.last_reconciled || "—");
}

function updateTemporalUi() {
  const now = serverNow();
  const interval = 5 * 60 * 1000;
  const elapsed = ((now % interval) + interval) % interval;
  const remaining = interval - elapsed;
  const minutes = Math.floor(remaining / 60000);
  const seconds = Math.floor((remaining % 60000) / 1000);
  setText("scan-countdown", `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`);
  $("scan-ring")?.style.setProperty("--scan-progress", `${elapsed / interval * 100}%`);
  setText("server-clock", `${new Date(now).toISOString().slice(11, 19)} UTC`);
  const streamAge = app.lastMessageAt
    ? Math.max(0, (Date.now() - app.lastMessageAt) / 1000)
    : null;
  setText(
    "runtime-freshness",
    streamAge === null
      ? "Connecting"
      : streamAge < 1
        ? "Live · <1s"
        : streamAge < 5
          ? `Live · ${streamAge.toFixed(1)}s`
          : `Stale · ${Math.floor(streamAge)}s`,
  );
  updateConnectionStatus();

  if (app.state?.prices) {
    for (const [symbol, quote] of Object.entries(app.state.prices)) {
      const card = app.quoteCards.get(symbol);
      if (!card) continue;
      const freshness = card.querySelector(".freshness");
      freshness.textContent = quote.updated_at ? ageLabel(quote.updated_at) : "live stream";
      freshness.classList.toggle("stale", quote.updated_at ? finite(ageSeconds(quote.updated_at), 999) > 5 : false);
    }
  }
  for (const item of document.querySelectorAll(".pipeline-meta time[data-timestamp]")) {
    item.textContent = ageLabel(item.dataset.timestamp);
  }
}

function openTradeManager(ticket) {
  const position = (app.state?.positions || []).find((item) => Number(item.ticket) === Number(ticket));
  if (!position) {
    toast("Position unavailable", "The position may have closed or refreshed.", "warning");
    return;
  }
  $("trade-ticket").value = position.ticket;
  app.managedPosition = position;
  setText("trade-title", `Manage ${position.symbol} · ${position.ticket}`);
  setText("trade-context", `${position.direction} ${finite(position.lot).toFixed(2)} lot · open ${price(position.open_price)} · current ${price(position.current_price)} · P/L ${signedMoney(position.profit_usd, app.state?.account?.currency || "USD")}`);
  $("trade-sl").value = finite(position.sl) > 0 ? position.sl : "";
  $("trade-tp").value = finite(position.tp) > 0 ? position.tp : "";
  updateProtectionEstimates();
  $("close-percent").value = 100;
  $("trade-dialog").showModal();
}

function exitValueRate(position, favorable) {
  const open = finite(position?.open_price);
  const current = finite(position?.current_price);
  const direction = String(position?.direction || "").toUpperCase() === "SELL" ? -1 : 1;
  const sl = finite(position?.sl);
  const tp = finite(position?.tp);
  const risk = finite(position?.risk_to_sl_usd);
  const reward = finite(position?.reward_to_tp_usd);
  const adverseDistance = Math.abs(open - sl);
  const favorableDistance = Math.abs(tp - open);
  const riskRate = adverseDistance > 0 && risk > 0 ? risk / adverseDistance : 0;
  const rewardRate = favorableDistance > 0 && reward > 0 ? reward / favorableDistance : 0;
  const liveDistance = Math.abs(current - open);
  const liveRate = liveDistance > 0 && Math.abs(finite(position?.profit_usd)) > 0
    ? Math.abs(finite(position.profit_usd)) / liveDistance
    : 0;
  if (favorable) return rewardRate || riskRate || liveRate;
  return riskRate || rewardRate || liveRate;
}

function updateProtectionEstimate(inputId, outputId) {
  const position = app.managedPosition;
  const output = $(outputId);
  if (!position || !output) return;
  const raw = $(inputId)?.value.trim() || "";
  const level = Number(raw);
  if (!raw || !Number.isFinite(level) || level <= 0) {
    output.textContent = "Estimated exit P/L —";
    colorValue(output, 0);
    return;
  }
  const open = finite(position.open_price);
  const direction = String(position.direction || "").toUpperCase() === "SELL" ? -1 : 1;
  const signedDistance = (level - open) * direction;
  const rate = exitValueRate(position, signedDistance >= 0);
  if (!(rate > 0)) {
    output.textContent = "Estimated exit P/L unavailable";
    colorValue(output, 0);
    return;
  }
  const estimate = signedDistance * rate;
  const currency = app.state?.account?.currency || "USD";
  output.textContent = `Estimated exit P/L ${signedMoney(estimate, currency)}`;
  colorValue(output, estimate);
}

function updateProtectionEstimates() {
  updateProtectionEstimate("trade-sl", "trade-sl-estimate");
  updateProtectionEstimate("trade-tp", "trade-tp-estimate");
}

async function openRejectedTradeReview(symbol, button) {
  await withPending(button, `rejected-preview-${symbol}`, async () => {
    try {
      const preview = await api(`/api/rejected/${encodeURIComponent(symbol)}/preview`);
      app.rejectedPreview = preview;
      setText("rejected-title", `Review ${preview.action} ${preview.symbol}`);
      setText("rejected-reason", preview.original_rejection || "Rejected by strategy gate");
      setText("rejected-order", `${preview.action} ${finite(preview.lot).toFixed(2)} lot`);
      setText("rejected-entry", price(preview.entry));
      setText("rejected-sl", price(preview.stop_loss));
      setText("rejected-tp", price(preview.take_profit));
      setText("rejected-risk", signedMoney(-Math.abs(finite(preview.risk_usd)), app.state?.account?.currency || "USD"));
      setText(
        "rejected-risk-cap",
        `${finite(preview.risk_pct).toFixed(2)}% · ${money(preview.execution_risk_ceiling_usd, app.state?.account?.currency || "USD")} final ceiling`,
      );
      setText("rejected-reward", signedMoney(Math.abs(finite(preview.reward_usd)), app.state?.account?.currency || "USD"));
      setText("rejected-rr", `${finite(preview.net_rr).toFixed(2)} R`);
      setText("rejected-margin", `${money(preview.margin_usd, app.state?.account?.currency || "USD")} · ${finite(preview.projected_margin_pct).toFixed(1)}% equity`);
      $("rejected-dialog").showModal();
    } catch (error) {
      toast("Rejected trade unavailable", error.message, "warning");
    }
  });
}

async function submitRejectedTrade(event) {
  event.preventDefault();
  const preview = app.rejectedPreview;
  if (!preview) return;
  const button = $("open-rejected-btn");
  await withPending(button, `rejected-open-${preview.symbol}`, async () => {
    const dialog = $("rejected-dialog");
    dialog.close();
    const confirmation = await requestConfirmation({
      title: `Force open ${preview.action} ${preview.symbol}`,
      message: `Submit one ${finite(preview.lot).toFixed(2)} lot market order while bypassing every project policy gate. Pepperstone account, quote, margin, SL/TP, and order checks remain mandatory.`,
      phrase: preview.confirmation_phrase,
      confirmLabel: "Force open trade",
    });
    if (!confirmation) {
      dialog.showModal();
      return;
    }
    try {
      const result = await api(`/api/rejected/${encodeURIComponent(preview.symbol)}/open`, {
        method: "POST",
        body: JSON.stringify({ confirmation }),
      });
      app.rejectedPreview = null;
      toast("Force order submitted", result.message || "Pepperstone verified the order.");
    } catch (error) {
      toast("Manual override blocked", error.message, "error");
    }
  });
}

function finishConfirmation(value) {
  const resolve = app.confirmResolve;
  app.confirmResolve = null;
  const dialog = $("confirm-dialog");
  if (dialog.open) dialog.close();
  if (resolve) resolve(value);
}

function requestConfirmation({ title, message, phrase, confirmLabel = "Confirm", danger = true }) {
  if (app.confirmResolve) finishConfirmation(null);
  setText("confirm-title", title);
  setText("confirm-message", message);
  setText("confirm-phrase", phrase);
  const input = $("confirm-input");
  input.value = "";
  const submit = $("confirm-submit");
  submit.textContent = confirmLabel;
  submit.className = `button ${danger ? "danger" : "primary"}`;
  submit.disabled = true;
  const dialog = $("confirm-dialog");
  dialog.showModal();
  requestAnimationFrame(() => input.focus());
  return new Promise((resolve) => { app.confirmResolve = resolve; });
}

async function toggleEngine() {
  const button = $("engine-btn");
  const running = Boolean(app.state?.engine_running);
  await withPending(button, "engine", async () => {
    try {
      const result = await api(running ? "/api/stop" : "/api/start", { method: "POST" });
      toast(running ? "Engine stopped" : "Engine started", result.message || "State will refresh from MT5.");
    } catch (error) {
      toast("Engine action failed", error.message, "error");
    }
  });
}

async function toggleArm() {
  const button = $("arm-btn");
  const automation = app.state?.automation || {};
  const account = app.state?.account || {};
  await withPending(button, "arm", async () => {
    try {
      if (automation.entries_armed) {
        const result = await api("/api/entries/disarm", { method: "POST" });
        toast("Entries disarmed", result.message || "New entries are blocked.");
        return;
      }
      const mode = automation.dry_run ? "PAPER" : account.account_mode || "DEMO";
      const phrase = automation.dry_run ? "ARM PAPER" : mode === "LIVE" ? "ARM LIVE" : "ARM DEMO";
      const confirmation = await requestConfirmation({
        title: `Arm ${mode} entries`,
        message: `Authorize new risk-approved entries for the active ${mode} MT5 account. Authorization is cleared after restart or account change.`,
        phrase,
        confirmLabel: "Arm entries",
        danger: mode === "LIVE",
      });
      if (!confirmation) return;
      const result = await api("/api/entries/arm", { method: "POST", body: JSON.stringify({ confirmation }) });
      toast("Entries armed", result.message || `Authorization bound to the active ${mode} account.`);
    } catch (error) {
      toast("Authorization failed", error.message, "error");
    }
  });
}

async function toggleAutonomy() {
  const button = $("autonomy-btn");
  const automation = app.state?.automation || {};
  const account = app.state?.account || {};
  await withPending(button, "autonomy", async () => {
    try {
      if (automation.autonomous_enabled) {
        const result = await api("/api/autonomy/disable", { method: "POST" });
        toast("Autonomy disabled", result.message || "Entries are disarmed.");
        return;
      }
      const mode = automation.dry_run ? "PAPER" : String(account.account_mode || "DEMO").toUpperCase();
      const phrase = `ENABLE AUTONOMOUS ${mode}`;
      const confirmation = await requestConfirmation({
        title: `Enable autonomous ${mode} trading`,
        message: `Persist entry authorization for this exact MT5 account (${mode}) across engine restarts. MT5 or model failures remain fail-closed, but this does not make the strategy profitable or eliminate losses.`,
        phrase,
        confirmLabel: "Enable autonomy",
        danger: mode === "LIVE",
      });
      if (!confirmation) return;
      const result = await api("/api/autonomy/enable", {
        method: "POST",
        body: JSON.stringify({ confirmation }),
      });
      toast("Autonomy enabled", result.message || `Bound to the active ${mode} account.`);
    } catch (error) {
      toast("Autonomy action failed", error.message, "error");
    }
  });
}

function optionalPositive(id) {
  const raw = $(id).value.trim();
  if (!raw) return null;
  const value = Number(raw);
  if (!Number.isFinite(value) || value <= 0) throw new Error("Protection prices must be positive numbers.");
  return value;
}

async function protectTrade() {
  const button = $("protect-btn");
  await withPending(button, "protect", async () => {
    try {
      const ticket = Number($("trade-ticket").value);
      const stopLoss = optionalPositive("trade-sl");
      const takeProfit = optionalPositive("trade-tp");
      if (stopLoss === null && takeProfit === null) throw new Error("Enter a stop-loss, take-profit, or both.");
      const live = app.state?.account?.account_mode === "LIVE" && !app.state?.automation?.dry_run;
      let confirmation = "";
      if (live) {
        const phrase = `PROTECT ${ticket}`;
        confirmation = await requestConfirmation({
          title: "Update live protection",
          message: "This submits new protection prices to Pepperstone for the selected live position.",
          phrase,
          confirmLabel: "Update protection",
        });
        if (!confirmation) return;
      }
      const result = await api(`/api/positions/${ticket}/protect`, {
        method: "POST",
        body: JSON.stringify({ stop_loss: stopLoss, take_profit: takeProfit, confirmation }),
      });
      $("trade-dialog").close();
      toast("Protection updated", result.message || "Pepperstone accepted the request.");
    } catch (error) {
      toast("Protection update failed", error.message, "error");
    }
  });
}

async function closeTrade() {
  const button = $("close-trade-btn");
  await withPending(button, "close-position", async () => {
    try {
      const ticket = Number($("trade-ticket").value);
      const percentageInput = $("close-percent");
      if (!percentageInput.reportValidity()) return;
      const percent = Number(percentageInput.value);
      const live = app.state?.account?.account_mode === "LIVE" && !app.state?.automation?.dry_run;
      let confirmation = "";
      if (live) {
        const phrase = `CLOSE ${ticket}`;
        confirmation = await requestConfirmation({
          title: `Close ${percent}% of live position`,
          message: "A live close is irreversible and can fill at a different price during fast markets.",
          phrase,
          confirmLabel: "Submit live close",
        });
        if (!confirmation) return;
      }
      const result = await api(`/api/positions/${ticket}/close`, {
        method: "POST",
        body: JSON.stringify({ percent, confirmation }),
      });
      $("trade-dialog").close();
      toast("Close submitted", result.message || "Waiting for broker reconciliation.");
    } catch (error) {
      toast("Close failed", error.message, "error");
    }
  });
}

async function loadConfig() {
  app.config = await api("/api/config");
  if (app.state) renderAll(app.state);
  return app.config;
}

function fillSettings(config) {
  $("c-symbols").value = (config.symbols || []).join(",");
  $("c-candidates").value = (config.market_candidate_symbols || config.symbols || []).join(",");
  $("c-dynamic-markets").value = String(Boolean(config.dynamic_market_selection_enabled));
  $("c-market-max").value = config.dynamic_market_max_symbols ?? 6;
  $("c-model-candidates").value = config.llm_entry_candidates_per_bar ?? 3;
  $("c-market-refresh").value = config.market_selection_refresh_seconds ?? 60;
  $("c-analysis").value = config.analysis_interval_seconds ?? 60;
  $("c-provider").value = config.llm_provider || "local";
  $("c-model").value = config.decision_model || config.local_llm_model || "";
  $("c-quantization").value = config.local_llm_required_quantization || "AUTO";
  $("c-reasoning").value = config.openai_reasoning_effort || "low";
  $("c-risk").value = config.risk_percent ?? 1;
  $("c-portfolio").value = config.max_portfolio_risk_pct ?? 3;
  $("c-maxpos").value = config.max_open_positions ?? 1;
  $("c-margin").value = config.max_margin_usage_pct ?? 35;
  $("c-rr").step = "0.01";
  $("c-rr").value = config.min_risk_reward_ratio ?? 1.47;
  $("c-confidence").value = config.confidence_threshold ?? .7;
  $("c-failed-reversal").value = String(
    config.failed_thesis_reversal_enabled ?? true
  );
  $("c-failed-reversal-confidence").value =
    config.failed_thesis_reversal_min_confidence ?? .65;
  $("c-failed-reversal-bars").value =
    config.failed_thesis_reversal_max_age_bars ?? 18;
  $("c-failed-reversal-m5-adx").value =
    config.failed_thesis_reversal_min_m5_adx ?? 25;
  $("c-failed-reversal-m15-adx").value =
    config.failed_thesis_reversal_min_m15_adx ?? 19.1;
  $("c-spread").value = config.max_spread_pips ?? 3;
  $("c-entry-adx").value = config.entry_min_adx ?? 25;
  $("c-adx-rising").value = String(config.entry_require_adx_rising ?? true);
  $("c-adx-decline").value = config.entry_adx_decline_tolerance ?? .5;
  $("c-aligned-adx-decline").value =
    config.entry_aligned_adx_decline_tolerance ?? 1.5;
  $("c-entry-max-extension").value = config.entry_max_candle_range_atr ?? 1.5;
  $("c-aligned-chase-confidence").value =
    config.entry_strong_alignment_chase_min_confidence ?? .85;
  $("c-aligned-chase-extension").value =
    config.entry_strong_alignment_chase_max_extension_atr ?? 2;
  $("c-breakout-displacement").value = config.breakout_min_displacement_atr ?? .1;
  $("c-zone-distance").value = config.entry_min_opposing_distance_atr ?? .75;
  $("c-unconfirmed-bos-zone").value = config.entry_unconfirmed_bos_min_opposing_distance_atr ?? 1.75;
  $("c-cost-extension").value = config.plan_max_cost_target_extension_r ?? .5;
  $("c-reentry-bars").value = config.same_thesis_reentry_min_bars ?? 2;
  $("c-profit-reentry-bars").value =
    config.same_thesis_profit_reentry_min_bars ?? 1;
  $("c-retest").value = String(config.retest_continuation_enabled ?? true);
  $("c-retest-min").value = config.retest_min_resumption_atr ?? .1;
  $("c-micro-profit").value = String(
    config.micro_profit_protection_enabled ?? false
  );
  $("c-shadow-symbols").value = (config.shadow_symbols || []).join(",");
  $("c-profit-on").value = String(Boolean(config.auto_close_profit_enabled));
  $("c-profit").value = config.auto_close_profit_usd ?? .1;
  $("c-loss-on").value = String(Boolean(config.auto_close_loss_enabled));
  $("c-loss").value = config.auto_close_loss_usd ?? .1;
  $("c-monitor-poll").value = config.decision_poll_seconds ?? .5;
  $("c-fast-exit").value = String(config.fast_exit_review_enabled ?? true);
  $("c-exit-timeframe").value = config.position_exit_review_timeframe || "M1";
  $("c-profit-lock").value = String(config.profit_lock_enabled ?? true);
  $("c-lock-trigger").value = config.profit_lock_trigger_r ?? .50;
  $("c-lock-floor").value = config.profit_lock_floor_usd ?? .08;
  $("c-final-lock-fallback").value = String(
    config.profit_lock_final_fallback_only ?? true
  );
  $("c-giveback").value = String(config.profit_giveback_enabled ?? true);
  $("c-giveback-trigger").value = config.profit_giveback_trigger_r ?? .5;
  $("c-giveback-close-min").value = config.profit_giveback_close_min_r ?? 1;
  $("c-giveback-fraction").value = config.profit_giveback_fraction ?? .5;
  $("c-breakeven-r").value = config.breakeven_trigger_r ?? .75;
  $("c-trailing-r").value = config.trailing_trigger_r ?? 1;
  $("c-trailing-distance").value = config.trailing_distance_r ?? .6;
  $("c-weekend").value = String(Boolean(config.weekend_trading_enabled));
  $("c-weekend-symbols").value = (config.weekend_symbols || []).join(",");
  syncDecisionProviderFields(false);
  syncObjectiveFields();
}

async function openSettings() {
  try {
    if (!app.config) await loadConfig();
    fillSettings(app.config);
    $("settings-dialog").showModal();
    await refreshLocalModelHealth();
  } catch (error) {
    toast("Settings unavailable", error.message, "error");
  }
}

function syncObjectiveFields() {
  $("c-profit").disabled = $("c-profit-on").value !== "true";
  $("c-loss").disabled = $("c-loss-on").value !== "true";
}

function syncDecisionProviderFields(resetModel = true) {
  const provider = $("c-provider").value;
  if (resetModel && app.config) {
    $("c-model").value = provider === "deterministic"
      ? "rules-v2-adaptive"
      : provider === "openai"
        ? (app.config.openai_model || "gpt-5.6-terra")
        : (app.config.local_llm_model || "");
  }
  $("c-model").disabled = provider === "deterministic";
  $("c-reasoning").disabled = provider !== "openai";
  $("c-quantization").disabled = provider !== "local";
  syncQuantizationProfile();
  setText(
    "c-provider-hint",
    provider === "deterministic"
      ? "No API, GPU inference, or model wait. Completed-candle rules still pass through every risk and broker gate."
      : provider === "openai"
      ? (app.config?.openai_configured
          ? "OpenAI key is configured. Changes require an engine restart."
          : "OPENAI_API_KEY is not configured; the engine will remain unavailable in this mode.")
      : "Local inference uses the configured LM Studio endpoint."
  );
}

function syncQuantizationProfile() {
  const quantization = $("c-quantization").value.trim().toUpperCase() || "AUTO";
  // Keep the project semaphore aligned with LM Studio's single loaded model
  // instance. Quantization changes memory/latency, not the safe parallelism
  // contract; hidden concurrency changes previously caused readiness failure.
  const concurrency = 1;
  const profileLabel = quantization === "AUTO"
    ? "follows any loaded quantization"
    : quantization;
  setText(
    "c-concurrency-profile",
    `${concurrency} prediction${concurrency === 1 ? "" : "s"} · ${profileLabel}`,
  );
  const health = app.localModelHealth || {};
  const loaded = health.quantization || "not detected";
  const matches = quantization === "AUTO"
    ? Boolean(health.loaded || health.available)
    : String(loaded).toUpperCase() === quantization;
  const status = matches
    ? "match"
    : quantization === "AUTO"
      ? "load the configured chat model in LM Studio"
      : "load the selected variant";
  setText(
    "c-quantization-hint",
    `LM Studio loaded: ${loaded} · selected: ${profileLabel} · ${status}`,
  );
}

async function refreshLocalModelHealth() {
  try {
    app.localModelHealth = await api("/api/llm/health");
  } catch (error) {
    app.localModelHealth = { quantization: "", error: error.message };
  }
  syncQuantizationProfile();
}

async function saveSettings(event) {
  event.preventDefault();
  const form = $("settings-form");
  if (!form.reportValidity()) return;
  const button = $("save-settings-btn");
  await withPending(button, "settings", async () => {
    const updates = {
      TRADING_SYMBOLS: $("c-symbols").value,
      MARKET_CANDIDATE_SYMBOLS: $("c-candidates").value,
      DYNAMIC_MARKET_SELECTION_ENABLED: $("c-dynamic-markets").value,
      DYNAMIC_MARKET_MAX_SYMBOLS: $("c-market-max").value,
      LLM_ENTRY_CANDIDATES_PER_BAR: $("c-model-candidates").value,
      MARKET_SELECTION_REFRESH_SECONDS: $("c-market-refresh").value,
      ANALYSIS_INTERVAL_SECONDS: $("c-analysis").value,
      LLM_PROVIDER: $("c-provider").value,
      LOCAL_LLM_REQUIRED_QUANTIZATION: $("c-quantization").value,
      OPENAI_REASONING_EFFORT: $("c-reasoning").value,
      RISK_PERCENT: $("c-risk").value,
      MAX_PORTFOLIO_RISK_PCT: $("c-portfolio").value,
      MAX_OPEN_POSITIONS: $("c-maxpos").value,
      MAX_MARGIN_USAGE_PCT: $("c-margin").value,
      MIN_RISK_REWARD_RATIO: $("c-rr").value,
      CONFIDENCE_THRESHOLD: $("c-confidence").value,
      FAILED_THESIS_REVERSAL_ENABLED: $("c-failed-reversal").value,
      FAILED_THESIS_REVERSAL_MIN_CONFIDENCE: $("c-failed-reversal-confidence").value,
      FAILED_THESIS_REVERSAL_MAX_AGE_BARS: $("c-failed-reversal-bars").value,
      FAILED_THESIS_REVERSAL_MIN_M5_ADX: $("c-failed-reversal-m5-adx").value,
      FAILED_THESIS_REVERSAL_MIN_M15_ADX: $("c-failed-reversal-m15-adx").value,
      MAX_SPREAD_PIPS: $("c-spread").value,
      ENTRY_MIN_ADX: $("c-entry-adx").value,
      ENTRY_REQUIRE_ADX_RISING: $("c-adx-rising").value,
      ENTRY_ADX_DECLINE_TOLERANCE: $("c-adx-decline").value,
      ENTRY_ALIGNED_ADX_DECLINE_TOLERANCE:
        $("c-aligned-adx-decline").value,
      ENTRY_MAX_CANDLE_RANGE_ATR: $("c-entry-max-extension").value,
      ENTRY_STRONG_ALIGNMENT_CHASE_MIN_CONFIDENCE:
        $("c-aligned-chase-confidence").value,
      ENTRY_STRONG_ALIGNMENT_CHASE_MAX_EXTENSION_ATR:
        $("c-aligned-chase-extension").value,
      BREAKOUT_MIN_DISPLACEMENT_ATR: $("c-breakout-displacement").value,
      ENTRY_MIN_OPPOSING_DISTANCE_ATR: $("c-zone-distance").value,
      ENTRY_UNCONFIRMED_BOS_MIN_OPPOSING_DISTANCE_ATR: $("c-unconfirmed-bos-zone").value,
      PLAN_MAX_COST_TARGET_EXTENSION_R: $("c-cost-extension").value,
      SAME_THESIS_REENTRY_MIN_BARS: $("c-reentry-bars").value,
      SAME_THESIS_PROFIT_REENTRY_MIN_BARS: $("c-profit-reentry-bars").value,
      RETEST_CONTINUATION_ENABLED: $("c-retest").value,
      RETEST_MIN_RESUMPTION_ATR: $("c-retest-min").value,
      MICRO_PROFIT_PROTECTION_ENABLED: $("c-micro-profit").value,
      SHADOW_SYMBOLS: $("c-shadow-symbols").value,
      AUTO_CLOSE_TARGET_PROFIT_ENABLED: $("c-profit-on").value,
      AUTO_CLOSE_TARGET_PROFIT_USD: $("c-profit").value || app.config.auto_close_profit_usd,
      AUTO_CLOSE_TARGET_LOSS_ENABLED: $("c-loss-on").value,
      AUTO_CLOSE_TARGET_LOSS_USD: $("c-loss").value || app.config.auto_close_loss_usd,
      DECISION_POLL_SECONDS: $("c-monitor-poll").value,
      FAST_EXIT_REVIEW_ENABLED: $("c-fast-exit").value,
      POSITION_EXIT_REVIEW_TIMEFRAME: $("c-exit-timeframe").value,
      PROFIT_LOCK_ENABLED: $("c-profit-lock").value,
      PROFIT_LOCK_TRIGGER_R: $("c-lock-trigger").value,
      PROFIT_LOCK_FLOOR_USD: $("c-lock-floor").value,
      PROFIT_LOCK_FINAL_FALLBACK_ONLY: $("c-final-lock-fallback").value,
      PROFIT_GIVEBACK_ENABLED: $("c-giveback").value,
      PROFIT_GIVEBACK_TRIGGER_R: $("c-giveback-trigger").value,
      PROFIT_GIVEBACK_CLOSE_MIN_R: $("c-giveback-close-min").value,
      PROFIT_GIVEBACK_FRACTION: $("c-giveback-fraction").value,
      BREAKEVEN_TRIGGER_R: $("c-breakeven-r").value,
      TRAILING_TRIGGER_R: $("c-trailing-r").value,
      TRAILING_DISTANCE_R: $("c-trailing-distance").value,
      WEEKEND_TRADING_ENABLED: $("c-weekend").value,
      WEEKEND_SYMBOLS: $("c-weekend-symbols").value,
    };
    if (updates.LLM_PROVIDER === "openai") {
      updates.OPENAI_MODEL = $("c-model").value;
    } else if (updates.LLM_PROVIDER === "local") {
      updates.LOCAL_LLM_MODEL = $("c-model").value;
    }
    try {
      const result = await api("/api/config", { method: "POST", body: JSON.stringify(updates) });
      // Display the effective in-process values. Persisted non-live settings
      // remain pending until restart and must not be shown as already active.
      app.config = await api("/api/config");
      $("settings-dialog").close();
      if (app.state) renderAll(app.state);
      toast(
        "Settings saved",
        result.restart_required
          ? "Saved to .env · restart the engine to apply non-live settings."
          : "The live-safe setting is active now.",
      );
    } catch (error) {
      toast("Settings not saved", error.message, "error");
    }
  });
}

function selectTab(name, focus = false) {
  const buttons = [...document.querySelectorAll('[role="tab"]')];
  for (const button of buttons) {
    const selected = button.dataset.tab === name;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-selected", String(selected));
    button.tabIndex = selected ? 0 : -1;
    const panel = $(`${button.dataset.tab}-tab`);
    if (panel) panel.hidden = !selected;
    if (selected && focus) button.focus();
  }
  $("history-count").hidden = name !== "history";
}

function bindTabs() {
  const tabs = [...document.querySelectorAll('[role="tab"]')];
  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => selectTab(tab.dataset.tab));
    tab.addEventListener("keydown", (event) => {
      let target = null;
      if (event.key === "ArrowRight") target = (index + 1) % tabs.length;
      if (event.key === "ArrowLeft") target = (index - 1 + tabs.length) % tabs.length;
      if (event.key === "Home") target = 0;
      if (event.key === "End") target = tabs.length - 1;
      if (target !== null) {
        event.preventDefault();
        selectTab(tabs[target].dataset.tab, true);
      }
    });
  });
}

function bindEvents() {
  $("engine-btn").addEventListener("click", toggleEngine);
  $("arm-btn").addEventListener("click", toggleArm);
  $("autonomy-btn").addEventListener("click", toggleAutonomy);
  $("settings-btn").addEventListener("click", openSettings);
  $("protect-btn").addEventListener("click", protectTrade);
  $("trade-sl").addEventListener("input", updateProtectionEstimates);
  $("trade-tp").addEventListener("input", updateProtectionEstimates);
  $("close-trade-btn").addEventListener("click", closeTrade);
  $("settings-form").addEventListener("submit", saveSettings);
  $("c-profit-on").addEventListener("change", syncObjectiveFields);
  $("c-loss-on").addEventListener("change", syncObjectiveFields);
  $("c-provider").addEventListener("change", () => syncDecisionProviderFields(true));
  $("c-quantization").addEventListener("change", syncQuantizationProfile);
  $("positions").addEventListener("click", (event) => {
    const button = event.target.closest("[data-manage-ticket]");
    if (button) openTradeManager(button.dataset.manageTicket);
  });
  $("symbol-pipeline").addEventListener("click", (event) => {
    const button = event.target.closest("[data-review-rejected]");
    if (button) openRejectedTradeReview(button.dataset.reviewRejected, button);
  });
  $("rejected-form").addEventListener("submit", submitRejectedTrade);
  document.addEventListener("click", (event) => {
    const button = event.target.closest("[data-close-dialog]");
    if (button) $(button.dataset.closeDialog)?.close();
  });

  $("confirm-input").addEventListener("input", () => {
    $("confirm-submit").disabled = $("confirm-input").value !== $("confirm-phrase").textContent;
  });
  $("confirm-form").addEventListener("submit", (event) => {
    event.preventDefault();
    if ($("confirm-input").value === $("confirm-phrase").textContent) finishConfirmation($("confirm-input").value);
  });
  $("confirm-cancel").addEventListener("click", () => finishConfirmation(null));
  $("confirm-dialog").addEventListener("cancel", (event) => {
    event.preventDefault();
    finishConfirmation(null);
  });
  $("confirm-dialog").addEventListener("close", () => {
    if (app.confirmResolve) finishConfirmation(null);
  });

  bindTabs();
  window.addEventListener("online", connect);
  window.addEventListener("offline", () => setStatus("s-ws", "offline", "NETWORK OFFLINE"));
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) return;
    if (!app.socket || app.socket.readyState === WebSocket.CLOSED) connect();
    // Background tabs can throttle animation frames while the broker keeps
    // trading. Always reconcile positions and account state when visible.
    refreshFullState();
  });
  window.addEventListener("beforeunload", () => {
    app.shuttingDown = true;
    app.socket?.close();
  });
}

async function boot() {
  bindEvents();
  connect();
  window.setInterval(updateTemporalUi, 500);
  try {
    await loadConfig();
  } catch (error) {
    toast("Configuration unavailable", error.message, "warning");
  }
  try {
    await refreshLocalModelHealth();
  } catch (error) {
    toast("Local model check failed", error.message, "warning");
  }
}

boot();
