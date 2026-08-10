import unittest
from pathlib import Path


class DashboardContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        static = Path(__file__).parents[1] / "ui" / "static"
        cls.html = (static / "index.html").read_text(encoding="utf-8")
        cls.css = (static / "terminal.css").read_text(encoding="utf-8")
        cls.js = (static / "terminal.js").read_text(encoding="utf-8")

    def test_market_watch_has_bid_and_ask_quotes(self):
        self.assertIn("Pepperstone stream", self.html)
        self.assertIn("Red SELL display", self.html)
        self.assertIn("Blue BUY display", self.html)
        self.assertIn("orders buy at ASK and sell at BID", self.html)
        self.assertIn(".quote-side.sell-side", self.css)
        self.assertIn(".quote-side.buy-side", self.css)
        self.assertIn('node("div", "side-label", "SELL")', self.js)
        self.assertIn('node("div", "side-label", "BUY")', self.js)
        # Requested presentation: red SELL shows the higher quote and blue BUY
        # shows the lower quote. This must not change MT5 execution semantics.
        self.assertIn('flashValue(card.querySelector(".sell-price"), quote.ask)', self.js)
        self.assertIn('flashValue(card.querySelector(".buy-price"), quote.bid)', self.js)

    def test_readiness_and_percentage_bars_are_present(self):
        for element_id in (
            "ready-state",
            "risk-progress",
            "margin-progress",
            "cpu-progress",
            "gpu-progress",
            "vram-progress",
            "ram-progress",
        ):
            self.assertIn(f'id="{element_id}"', self.html)

        self.assertIn("GPU Performance", self.html)
        self.assertIn("VRAM Usage", self.html)
        self.assertIn('setProgress("gpu-progress", gpu)', self.js)
        self.assertIn('setProgress("vram-progress", vram)', self.js)

    def test_external_assets_are_declared(self):
        self.assertRegex(self.html, r'href="/static/terminal\.css(?:\?[^\"]+)?"')
        self.assertRegex(self.html, r'src="/static/terminal\.js(?:\?[^\"]+)?"')

    def test_position_manager_displays_dollar_exit_estimates(self):
        self.assertIn('id="trade-sl-estimate"', self.html)
        self.assertIn('id="trade-tp-estimate"', self.html)
        self.assertIn("function updateProtectionEstimates()", self.js)

    def test_minimum_risk_reward_supports_hundredths(self):
        self.assertIn('$("c-rr").step = "0.01"', self.js)
        self.assertIn(
            'config.min_risk_reward_ratio ?? 1.47',
            self.js,
        )

    def test_entry_noise_and_cost_adjustment_controls_are_present(self):
        for element_id in (
            "c-adx-decline",
            "c-breakout-displacement",
            "c-cost-extension",
        ):
            self.assertIn(f'id="{element_id}"', self.html)
        self.assertIn("ENTRY_ADX_DECLINE_TOLERANCE", self.js)
        self.assertIn("BREAKOUT_MIN_DISPLACEMENT_ATR", self.js)
        self.assertIn("PLAN_MAX_COST_TARGET_EXTENSION_R", self.js)

    def test_retest_and_optional_twenty_cent_controls_are_present(self):
        for element_id in ("c-retest", "c-retest-min", "c-micro-profit"):
            self.assertIn(f'id="{element_id}"', self.html)
        self.assertIn("RETEST_CONTINUATION_ENABLED", self.js)
        self.assertIn("RETEST_MIN_RESUMPTION_ATR", self.js)
        self.assertIn("MICRO_PROFIT_PROTECTION_ENABLED", self.js)
        self.assertIn(
            'config.micro_profit_protection_enabled ?? false',
            self.js,
        )

    def test_model_switch_keeps_single_local_inference_slot(self):
        self.assertIn('const concurrency = 1;', self.js)
        self.assertIn('LLM_MAX_CONCURRENCY: "1"', self.js)
        self.assertNotIn('quantization === "Q6_K" ? 2 : 1', self.js)

    def test_shadow_outcome_summary_is_visible(self):
        self.assertIn('id="shadow-summary"', self.html)
        self.assertIn("function renderShadowEvidence", self.js)
        self.assertIn("Diagnostic only", self.js)

    def test_live_rendering_is_coalesced_and_heavy_sections_are_cached(self):
        self.assertIn("scheduleDashboardRender", self.js)
        self.assertIn("renderWhenChanged", self.js)
        self.assertIn("app.quoteCards.get(symbol)", self.js)
        self.assertNotIn("void element.offsetWidth", self.js)

    def test_position_state_has_independent_freshness_recovery(self):
        self.assertIn("function refreshFullState()", self.js)
        self.assertIn("Date.now() - app.lastFullStateAt", self.js)
        self.assertIn('setStatus("s-ws", "warning", "SYNCING STATE")', self.js)
        self.assertIn("refreshFullState();", self.js)
        self.assertIn("if (document.hidden) return;", self.js)

    def test_rejected_trade_override_requires_review_and_confirmation(self):
        self.assertIn('id="rejected-dialog"', self.html)
        self.assertIn('id="open-rejected-btn"', self.html)
        self.assertIn('id="rejected-risk-cap"', self.html)
        self.assertIn("decision.manual_override_available", self.js)
        self.assertIn("preview.confirmation_phrase", self.js)
        self.assertIn("Force open at broker minimum", self.html)
        self.assertIn("execution_risk_ceiling_usd", self.js)
        self.assertIn("/api/rejected/", self.js)
        self.assertIn("payload?.reason", self.js)

    def test_autonomous_mode_is_explicit_and_account_bound(self):
        self.assertIn('id="autonomy-btn"', self.html)
        self.assertIn("ENABLE AUTONOMOUS", self.js)
        self.assertIn("/api/autonomy/enable", self.js)
        self.assertIn("/api/autonomy/disable", self.js)
        self.assertIn("exact MT5 account", self.js)

    def test_unattended_loop_stall_is_visible_and_watched(self):
        watchdog = (
            Path(__file__).parents[1] / "run_unattended.ps1"
        ).read_text(encoding="utf-8")
        self.assertIn('LOOP: "ENGINE STALLED"', self.js)
        self.assertIn('"LOOP"', self.js)
        self.assertIn("broker_poll_heartbeat_utc", watchdog)
        self.assertIn("MaxHeartbeatAgeSeconds", watchdog)


if __name__ == "__main__":
    unittest.main()
