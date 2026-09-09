import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import unittest

from risk.instruments import instrument_asset_class
from ui.state import DashboardState


class WatchDataDisplayTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node is required for executable dashboard tests")
    def test_live_candidate_is_not_approval_and_expires_after_two_minutes(self):
        source = Path(__file__).parents[1] / "ui" / "static" / "terminal.js"
        script = r'''
const fs = require('fs'), vm = require('vm'), context = {};
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8').replace(/boot\(\);\s*$/, ''), context);
const now = Date.parse('2032-03-01T12:30:00Z');
const live = {eligible:true,direction:'BUY',signal_time_utc:'2032-03-01T12:30:00Z'};
const label = {hidden:false,textContent:'BUY reversal LIVE CANDIDATE - not approved',dataset:{researchSignal:live.signal_time_utc}};
context.document = {querySelectorAll:()=>[label]};
context.expireResearchLabels(now+121000);
console.log(JSON.stringify([context.researchWatchLabel({live_reversal:live},now),
context.researchWatchLabel({live_reversal:live},now+121000),label.hidden]));
'''
        result = subprocess.run([shutil.which("node"), "-e", script, str(source)], capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        self.assertIn("LIVE CANDIDATE - not approved", data[0])
        self.assertEqual(data[1:], ["", True])

    @unittest.skipUnless(shutil.which("node"), "Node is required for executable dashboard tests")
    def test_cached_pipeline_labels_expire_and_missing_quotes_stay_unavailable(self):
        source = Path(__file__).parents[1] / "ui" / "static" / "terminal.js"
        script = r'''
const fs = require('fs'), vm = require('vm');
const context = {};
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8').replace(/boot\(\);\s*$/, ''), context);
const labels = [
  {hidden:false,dataset:{researchSignal:'2032-03-01T12:20:00Z'}},
  {hidden:false,dataset:{researchSignal:'2032-03-01T12:25:00Z'}},
  {hidden:false,dataset:{researchSignal:'2032-03-01T12:35:00Z'}},
];
context.document = {querySelectorAll:()=>labels};
context.expireResearchLabels(Date.parse('2032-03-01T12:26:00Z'));
console.log(JSON.stringify({hidden:labels.map(item=>item.hidden),fresh:context.quoteFreshness({}, {status:'HISTORY UNAVAILABLE',reason:'No broker history'})}));
'''
        result = subprocess.run([shutil.which("node"), "-e", script, str(source)], capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        self.assertEqual(data["hidden"], [True, False, True])
        self.assertTrue(data["fresh"]["stale"])
        self.assertEqual(data["fresh"]["text"], "HISTORY UNAVAILABLE")

    @unittest.skipUnless(shutil.which("node"), "Node is required for executable dashboard tests")
    def test_research_watch_is_labeled_non_live_and_expires(self):
        source = Path(__file__).parents[1] / "ui" / "static" / "terminal.js"
        script = r'''
const fs = require('fs'), vm = require('vm');
const context = {};
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8').replace(/boot\(\);\s*$/, ''), context);
const now = Date.parse('2032-03-01T12:30:10Z');
const watch = {candidate:true,live_eligible:false,direction:'BUY',signal_time_utc:'2032-03-01T12:30:00Z'};
console.log(JSON.stringify([
  context.researchWatchLabel({research_watch:watch},now),
  context.researchWatchLabel({research_watch:watch},now+300000),
  context.researchWatchLabel({research_watch:watch},now-20000),
  context.researchWatchLabel({research_watch:{...watch,live_eligible:true}},now),
  context.researchWatchLabel({},now),
]));
'''
        result = subprocess.run([shutil.which("node"), "-e", script, str(source)], capture_output=True, text=True, check=True)
        rows = json.loads(result.stdout)
        self.assertIn("RESEARCH ONLY", rows[0])
        self.assertEqual(rows[1:], ["", "", "", ""])

    @unittest.skipUnless(shutil.which("node"), "Node is required for executable dashboard tests")
    def test_missing_values_and_real_zero_spread_are_distinguished_in_javascript(self):
        source = Path(__file__).parents[1] / "ui" / "static" / "terminal.js"
        script = r'''
const fs = require('fs');
const vm = require('vm');
const context = {};
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8').replace(/boot\(\);\s*$/, ''), context);
const samples = [
  [{}, {status:'HISTORY STALE',selection_regime:'UNKNOWN',asset_class:'STOCK/CFD'}],
  [{bid:1,ask:1,spread_value:0,adx:null}, {}],
  [{bid:1.1,ask:1,spread_value:0,adx:null}, {}],
  [{bid:1,ask:1.01,spread_value:1,adx:25}, {selection_adx:36.8,selection_regime:'CONFIRMED_BEARISH'}],
  [{bid:null,ask:null,spread_value:null,adx:null}, {}],
];
console.log(JSON.stringify(samples.map(([q,f]) => context.watchCardMetrics(q,f))));
'''
        result = subprocess.run([shutil.which("node"), "-e", script, str(source)], capture_output=True, text=True, check=True)
        rows = json.loads(result.stdout)
        self.assertIsNone(rows[0]["spread"])
        self.assertIsNone(rows[0]["adx"])
        self.assertEqual(rows[0]["trend"], "NOT ANALYZED")
        self.assertEqual(rows[0]["assetClass"], "STOCK/CFD")
        self.assertEqual(rows[1]["spread"], 0)
        self.assertFalse(rows[2]["hasQuote"])
        self.assertIsNone(rows[2]["spread"])
        self.assertEqual(rows[3]["adx"], 36.8)
        self.assertIsNone(rows[4]["spread"])

    def test_tick_only_update_does_not_invent_an_adx_reading(self):
        state = DashboardState()
        state.update_prices("TEST", 100, 100.1, 0.01)
        self.assertIsNone(state.prices["TEST"].adx)

    def test_asset_labels_come_from_broker_contract_metadata(self):
        for path, expected in (("Markets/Stocks/USA", "STOCK/CFD"),
                               ("Markets/ETFs", "ETF/CFD"),
                               ("Markets/Indices", "INDEX/CFD"),
                               ("Markets/Forex", "FX/CFD"),
                               ("Markets/Crypto", "CRYPTO")):
            self.assertEqual(instrument_asset_class("TEST", SimpleNamespace(path=path)), expected)
