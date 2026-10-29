#!/usr/bin/env python
"""QAD pilot progress GUI - zero-dependency (stdlib http.server).

Serves a live progress page at http://localhost:8765 showing:
  - overall progress bar across phases (joint 1000, chained s1 1000,
    chained precompute 16 batches, chained s2 1000, merge checks, 6 ppl runs)
  - current phase, latest metrics, elapsed time
  - checkpoint files written so far
  - tail of the raw pilot log
Reads the pilot log (default analysis/qad/pilot5.log); page auto-refreshes 5 s.
"""
import json, re, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ADIR = Path(__file__).parent
LOG = ADIR / "pilot5.log"
CKPT_DIRS = [ADIR / "adapters_joint_ckpt", ADIR / "adapters_chained_ckpt"]
PORT = 8765

UPDATES = 1000
PHASES = [
    ("joint arm", UPDATES),
    ("chained s1 (L0)", UPDATES),
    ("chained precompute", 16),
    ("chained s2 (L63)", UPDATES),
    ("merge/pin checks", 1),
    ("ppl runs (6)", 6),
]
TOTAL_UNITS = sum(u for _, u in PHASES)

RE_JOINT = re.compile(r"joint step (\d+) loss ([\d.]+) mse0 ([\d.]+) mse63 ([\d.]+) \((\d+)s\)")
RE_S1 = re.compile(r"chained s1 step (\d+) mse0 ([\d.]+) \((\d+)s\)")
RE_S2 = re.compile(r"chained s2 step (\d+) mse63 ([\d.]+) \((\d+)s\)")
RE_PRE = re.compile(r"precompute batch (\d+)/16")
RE_PPL = re.compile(r"ppl\[([^\]]+)\] (\d+)/(\d+) windows")
RE_PPL_FINAL = re.compile(r"ppl\[([^\]]+)\] FINAL=([\d.]+)")
RE_DONE = re.compile(r"wrote .*pilot_results.json|PPL summary")


def read_log():
    if not LOG.exists():
        return [], f"log not found yet: {LOG.name}"
    try:
        with open(LOG, "r", errors="replace") as f:
            return f.readlines(), None
    except Exception as e:
        return [], str(e)


def parse(lines):
    state = {"joint": 0, "s1": 0, "s2": 0, "pre": 0, "ppl": 0, "checks": 0}
    info = {}
    for ln in lines[-400:]:
        m = RE_JOINT.search(ln)
        if m:
            state["joint"] = int(m[1])
            info["latest"] = f"joint step {m[1]} loss {m[2]} mse0 {m[3]} mse63 {m[4]} ({m[5]}s)"
        m = RE_S1.search(ln)
        if m:
            state["s1"] = int(m[1])
            info["latest"] = f"chained s1 step {m[1]} mse0 {m[2]} ({m[3]}s)"
        m = RE_S2.search(ln)
        if m:
            state["s2"] = int(m[1])
            info["latest"] = f"chained s2 step {m[1]} mse63 {m[2]} ({m[3]}s)"
        m = RE_PRE.search(ln)
        if m:
            state["pre"] = int(m[1])
            info["latest"] = f"chained precompute batch {m[1]}/16"
        m = RE_PPL.search(ln)
        if m:
            state["ppl"] = max(state["ppl"], int(m[3]) // max(int(m[2]), 1))
            info["latest"] = f"ppl[{m[1]}] window {m[2]}/{m[3]}"
        m = RE_PPL_FINAL.search(ln)
        if m:
            info[f"ppl {m[1]}"] = m[2]
        if "merge_check" in ln or "pin_check" in ln:
            state["checks"] = 1
        if RE_DONE.search(ln):
            state["ppl"] = 6
            info["status"] = "COMPLETE - pilot_results.json written"
    phase_idx, units = 0, 0
    if state["joint"] > 0:
        phase_idx, units = 0, state["joint"]
    if state["s1"] > 0:
        phase_idx, units = 1, state["s1"]
    if state["pre"] > 0:
        phase_idx, units = 2, state["pre"]
    if state["s2"] > 0:
        phase_idx, units = 3, state["s2"]
    if state["checks"] and state["s2"] >= UPDATES:
        phase_idx, units = 4, 1
    if state["ppl"] > 0:
        phase_idx, units = 5, state["ppl"]
    done = sum(u for _, u in PHASES[:phase_idx]) + units
    elapsed = ""
    m = re.search(r"\((\d+)s\)", lines[-1]) if lines else None
    if m:
        s = int(m[1])
        elapsed = f"{s // 3600}h{(s % 3600) // 60:02d}m{s % 60:02d}s"
    ckpts = []
    for d in CKPT_DIRS:
        if d.exists():
            ckpts += [f"{d.name}/{p.name}" for p in sorted(d.glob("*.safetensors"))[-6:]]
    return {
        "phase": PHASES[phase_idx][0] if lines else "waiting for log...",
        "phase_idx": phase_idx,
        "units_done": units,
        "phase_total": PHASES[phase_idx][1],
        "done": done,
        "total": TOTAL_UNITS,
        "elapsed": elapsed,
        "info": info,
        "checkpoints": ckpts,
        "log_tail": [l.rstrip() for l in lines[-40:]],
    }


PAGE = """<!doctype html><html><head><meta charset=utf-8>
<title>QAD Pilot progress</title><meta http-equiv=refresh content=5>
<style>
body{font-family:Segoe UI,system-ui,sans-serif;background:#101418;color:#d8dee9;margin:2em}
h1{font-size:1.2em;font-weight:600}
.bar{height:26px;background:#2a3140;border-radius:6px;overflow:hidden;margin:1em 0}
.bar>div{height:100%%;background:#4c9aff;transition:width 1s}
.pct{font-size:1.4em;font-weight:700}
.info{font-family:Consolas,monospace;font-size:.95em;line-height:1.6}
.ckpt{color:#8aa;font-size:.85em;margin-top:1em}
pre{background:#1a2029;padding:1em;border-radius:6px;font-size:.8em;max-height:20em;overflow:auto}
</style></head><body>
<h1>QAD Pilot - %(phase)s</h1>
<div class=pct>%(pct)s</div>
<div class=bar><div style="width:%(pct)s"></div></div>
<div class=info>%(info)s</div>
<div class=ckpt>checkpoints: %(ckpts)s</div>
<h2>log tail</h2><pre>%(tail)s</pre>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/stats"):
            lines, err = read_log()
            data = parse(lines) if not err else {"error": err, "log_tail": []}
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            lines, err = read_log()
            if err:
                html = f"<h1>waiting: {err}</h1>"
            else:
                d = parse(lines)
                pct = f"{100.0 * d['done'] / d['total']:.1f}%"
                info = " | ".join(f"<b>{k}</b>: {v}" for k, v in d["info"].items()) + \
                    f" | elapsed <b>{d['elapsed'] or '—'}</b>"
                html = PAGE % {
                    "phase": d["phase"], "pct": pct, "info": info,
                    "ckpts": ", ".join(d["checkpoints"]) or "(none yet)",
                    "tail": "\n".join(d["log_tail"]),
                }
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    if len(sys.argv) > 1:
        LOG = Path(sys.argv[1])
    srv = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"serving http://localhost:{PORT} log={LOG}", flush=True)
    srv.serve_forever()
