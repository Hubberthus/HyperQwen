#!/usr/bin/env python
"""QAD pilot progress GUI.

Tails a pilot log file (default: analysis/qad/pilot5.log) and shows:
  - overall progress bar across all phases (joint 1000, chained s1 1000,
    chained s2 precompute 16 batches + 1000 steps, 6 ppl runs, summary)
  - current phase, step, latest loss/mse metrics, elapsed, ETA
  - checkpoint files written so far
Launches inside WSL (WSLg display) or any desktop with tkinter.
"""
import json, re, sys, time
from pathlib import Path
import tkinter as tk
from tkinter import ttk, scrolledtext

ADIR = Path(__file__).parent
LOG = ADIR / "pilot5.log"
CKPT_DIRS = [ADIR / "adapters_joint_ckpt", ADIR / "adapters_chained_ckpt"]

UPDATES = 1000
PHASES = [  # (name, total units)
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


class PilotGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("QAD Pilot — Phase 2+3 progress")
        self.geometry("640x480")
        self.phase_idx = 0
        self.phase_units = 0
        self.last_line_no = 0
        self.metrics = {}

        head = ttk.Frame(self); head.pack(fill="x", padx=10, pady=8)
        self.phase_lbl = ttk.Label(head, text="phase: waiting for log…", font=("", 12, "bold"))
        self.phase_lbl.pack(anchor="w")
        self.bar = ttk.Progressbar(self, maximum=TOTAL_UNITS, value=0)
        self.bar.pack(fill="x", padx=10, pady=4)
        self.info = ttk.Label(head, text="", font=("TkFixedFont", 10), justify="left")
        self.info.pack(anchor="w", pady=4)
        self.ckpt = ttk.Label(head, text="checkpoints: (none yet)", foreground="#555")
        self.ckpt.pack(anchor="w")

        self.txt = scrolledtext.ScrolledText(self, height=14, font=("TkFixedFont", 9))
        self.txt.pack(fill="both", expand=True, padx=10, pady=8)
        self.after(1000, self.tick)

    def set_phase(self, idx, units_done):
        if idx != self.phase_idx:
            self.phase_idx = idx
        self.bar["value"] = sum(u for _, u in PHASES[:idx]) + units_done
        done = self.bar["value"]
        pct = 100.0 * done / TOTAL_UNITS
        self.phase_lbl["text"] = f"phase: {PHASES[idx][0]}  ({done}/{TOTAL_UNITS} units, {pct:.1f}% of run)"

    def tick(self):
        try:
            if LOG.exists():
                with open(LOG, "r", errors="replace") as f:
                    lines = f.readlines()
                self.parse(lines)
                if self.last_line_no != len(lines):
                    new = lines[self.last_line_no:]
                    self.last_line_no = len(lines)
                    for ln in new[-200:]:
                        self.txt.insert("end", ln.rstrip() + "\n")
                        self.txt.see("end")
            names = []
            for d in CKPT_DIRS:
                if d.exists():
                    names += [f"{d.name}/{p.name}" for p in sorted(d.glob("*.safetensors"))[-6:]]
            self.ckpt["text"] = "checkpoints: " + (", ".join(names) if names else "(none yet)")
        except Exception as e:
            self.info["text"] = f"gui error: {e}"
        self.after(1000, self.tick)

    def parse(self, lines):
        state = {"joint": 0, "s1": 0, "s2": 0, "pre": 0, "ppl": 0, "checks": 0}
        info = {}
        for ln in lines[-400:]:
            m = RE_JOINT.search(ln)
            if m:
                state["joint"] = int(m[1]); info["latest"] = f"step {m[1]} loss {m[2]} mse0 {m[3]} mse63 {m[4]} ({m[5]}s)"
            m = RE_S1.search(ln)
            if m:
                state["s1"] = int(m[1]); info["latest"] = f"chained s1 step {m[1]} mse0 {m[2]} ({m[3]}s)"
            m = RE_S2.search(ln)
            if m:
                state["s2"] = int(m[1]); info["latest"] = f"chained s2 step {m[1]} mse63 {m[2]} ({m[3]}s)"
            m = RE_PRE.search(ln)
            if m:
                state["pre"] = int(m[1])
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
                state["ppl"] = 6; info["status"] = "COMPLETE — pilot_results.json written"
        # pick deepest phase reached
        if state["joint"] > 0:
            self.set_phase(0, state["joint"])
        if state["s1"] > 0:
            self.set_phase(1, state["s1"])
        if state["pre"] > 0:
            self.set_phase(2, state["pre"])
        if state["s2"] > 0:
            self.set_phase(3, state["s2"])
        if state["checks"] > 0 and state["s2"] >= UPDATES:
            self.set_phase(4, 1)
        if state["ppl"] > 0:
            self.set_phase(5, state["ppl"])
        info_lines = [f"{k}: {v}" for k, v in info.items()]
        elapsed = ""
        m = re.search(r"\((\d+)s\)$", lines[-1].strip()) if lines else None
        if m:
            s = int(m[1]); elapsed = f" | elapsed {s//3600}h{(s%3600)//60:02d}m"
        self.info["text"] = " | ".join(info_lines[-2:] + [elapsed])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        LOG = Path(sys.argv[1])
    PilotGUI().mainloop()
