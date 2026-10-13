"""watch_gui.py — progress monitor for the salted-quantization FP16-originals
probe (docs/superpowers/plans/2026-10-12-salted-quantization-fp16-originals.md).

Stdlib-only tkinter app. Run on Windows:

    python analysis\\salted\\fp16\\watch_gui.py

Watches, every 3 s:
  - models-fp16/Qwen3.8-27B/ over the \\\\wsl.localhost UNC path: completed
    bytes + in-flight (.cache *.incomplete) vs expected.json
  - the WSL process table (wsl.exe ps): live probe scripts, pid + elapsed
  - tee'd logs analysis/salted/fp16/<script>.log: tail + parsed progress
  - artifacts (fetch.json / regression_gate.json / heterogeneity.json /
    codebook_gain.json) and the findings doc: task completion + key numbers

`python watch_gui.py --once` does one refresh, prints state as JSON, exits
(headless smoke test, no window).
"""

import json
import os
import re
import subprocess
import sys
import time

REPO = r"\\wsl.localhost\Ubuntu-24.04\home\hubbe\qwen-serving"
FP16 = os.path.join(REPO, "analysis", "salted", "fp16")
SALTED = os.path.join(REPO, "analysis", "salted")
DEST = os.path.join(REPO, "models-fp16", "Qwen3.8-27B")
FINDINGS = os.path.join(REPO, "docs", "superpowers", "plans",
                        "2026-10-12-salted-quantization-fp16-findings.md")

POLL_MS = 3000

TASKS = [
    ("t0", "Task 0  fetch originals Qwen/Qwen3.8-27B (~55.6 GB)", 0.30),
    ("t1", "Task 1  loader swap + regression gate (1e-12 vs baseline)", 0.15),
    ("t2", "Task 2  eta^2 heterogeneity on originals (9 tensors x 20 salts)", 0.20),
    ("t3", "Task 3  codebook gain on originals (9 tensors x 3 bits x 3 arms)", 0.25),
    ("t4", "Task 4  findings doc + SDD ledger", 0.10),
]

# WSL cmdline fragment -> (task id, log file). Order matters: the --regression
# fragment must be tested before the bare codebook_gain_fp16.py fragment.
SCRIPT_MAP = [
    ("fetch_originals.py", "t0", "fetch_originals.log"),
    ("codebook_gain_fp16.py --regression", "t1", "codebook_gain_fp16_regression.log"),
    ("heterogeneity_fp16.py", "t2", "heterogeneity_fp16.log"),
    ("codebook_gain_fp16.py", "t3", "codebook_gain_fp16.log"),
]

STATUS_STYLE = {
    "done":    ("DONE", "#c8e6c9", "#1b5e20"),
    "running": ("RUN ", "#bbdefb", "#0d47a1"),
    "pending": ("PEND", "#eeeeee", "#616161"),
    "failed":  ("FAIL", "#ffcdd2", "#b71c1c"),
    "skipped": ("SKIP", "#fff9c4", "#f57f17"),
}


def read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def scan_dest(expected):
    """Download progress: completed bytes + in-flight (.cache incomplete)."""
    exp = {f["path"]: f["size"] for f in (expected or {}).get("files", [])}
    total_exp = (expected or {}).get("total_bytes", 0)
    n_exp = (expected or {}).get("n_files", 0)
    done_bytes = inflight = 0
    n_done = 0
    current = None
    if os.path.isdir(DEST):
        for root, _dirs, files in os.walk(DEST):
            for name in files:
                p = os.path.join(root, name)
                rel = os.path.relpath(p, DEST).replace(os.sep, "/")
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                if current is None or st.st_mtime > current[1]:
                    current = (rel, st.st_mtime)
                if ".cache" in rel.split("/"):
                    if name.endswith(".incomplete"):
                        inflight += st.st_size
                    continue
                done_bytes += st.st_size
                if exp.get(rel) == st.st_size:
                    n_done += 1
    return {"expected_bytes": total_exp, "expected_files": n_exp,
            "done_bytes": done_bytes, "inflight_bytes": inflight,
            "done_files": n_done, "current_file": current}


def ps_probe_scripts():
    """Live probe processes in WSL -> {task_id: {pid, elapsed_s, cmd}}."""
    try:
        r = subprocess.run(
            ["wsl.exe", "-d", "Ubuntu-24.04", "--", "bash", "-c",
             "ps -eo pid,etime,args"],
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return {}
    found = {}
    for line in r.stdout.splitlines():
        if "analysis/salted/fp16" not in line or "venv/bin/python" not in line:
            continue
        m = re.match(r"\s*(\d+)\s+(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+)\s+(.*)", line)
        if not m:
            continue
        elapsed = (int(m.group(2) or 0) * 86400 + int(m.group(3) or 0) * 3600
                   + int(m.group(4)) * 60 + int(m.group(5)))
        cmd = m.group(6)
        for frag, tid, _log in SCRIPT_MAP:
            if frag in cmd:
                found[tid] = {"pid": int(m.group(1)), "elapsed_s": elapsed,
                              "cmd": cmd}
                break
    return found


def tail(path, max_bytes=8000):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return None


def gather_state():
    st = {"time": time.strftime("%H:%M:%S")}
    expected = read_json(os.path.join(FP16, "expected.json"))
    st["dest"] = scan_dest(expected)
    st["procs"] = ps_probe_scripts()

    fetch = read_json(os.path.join(FP16, "fetch.json"))
    reg = read_json(os.path.join(FP16, "regression_gate.json"))
    het = read_json(os.path.join(FP16, "heterogeneity.json"))
    het_base = read_json(os.path.join(SALTED, "heterogeneity.json"))
    cbg = read_json(os.path.join(FP16, "codebook_gain.json"))
    cbg_base = read_json(os.path.join(SALTED, "codebook_gain.json"))
    st["findings_exists"] = os.path.exists(FINDINGS)

    tasks = {}
    logs = {}

    # ---- Task 0: fetch
    t = {"id": "t0", "label": TASKS[0][1], "weight": TASKS[0][2]}
    d = st["dest"]
    if fetch:
        tc = fetch.get("tensor_check", {})
        dtype = next((v.get("dtype_on_disk") for v in tc.values()), "?")
        corrs = [f"{v['corr']:.4f}" for v in fetch.get("cross_check", {}).values()]
        t.update(state="done", frac=1.0,
                 note=(f"verdict={fetch.get('verdict')} dtype={dtype} "
                       f"shards={fetch.get('shards_on_disk')} "
                       f"corr(L27 q/gate/down)={', '.join(corrs)} "
                       f"download {fetch.get('download_wall_s')}s"))
    elif "t0" in st["procs"] or d["done_bytes"] + d["inflight_bytes"] > 0:
        frac = ((d["done_bytes"] + d["inflight_bytes"]) / d["expected_bytes"]
                if d["expected_bytes"] else 0.0)
        t.update(state="running", frac=frac,
                 note=(f"{d['done_files']}/{d['expected_files']} files, "
                       f"{(d['done_bytes'] + d['inflight_bytes']) / 1e9:.1f}/"
                       f"{d['expected_bytes'] / 1e9:.1f} GB"
                       + (f", now: {d['current_file'][0]}"
                          if d["current_file"] else "")))
    else:
        t.update(state="pending", frac=0.0, note="")
    tasks["t0"] = t
    logs["t0"] = tail(os.path.join(FP16, "fetch_originals.log"))

    # ---- Task 1: regression gate
    t = {"id": "t1", "label": TASKS[1][1], "weight": TASKS[1][2]}
    if reg and reg.get("passed"):
        t.update(state="done", frac=1.0,
                 note=(f"PASSED wall={reg.get('wall_s')}s — all baseline cells "
                       f"reproduced at 1e-12"))
    elif "t1" in st["procs"]:
        t.update(state="running", frac=0.5, note="sweeping 9 baseline tensors")
    else:
        t.update(state="pending", frac=0.0, note="")
    tasks["t1"] = t
    logs["t1"] = tail(os.path.join(FP16, "codebook_gain_fp16_regression.log"))

    # ---- Task 2: heterogeneity on originals
    t = {"id": "t2", "label": TASKS[2][1], "weight": TASKS[2][2]}
    log = tail(os.path.join(FP16, "heterogeneity_fp16.log"))
    logs["t2"] = log
    if het:
        t.update(state="done", frac=1.0, note="wrote fp16/heterogeneity.json")
    elif "t2" in st["procs"]:
        n = len(re.findall(r"^(?:layers\.\S+|lm_head)\s+\S+\s+[0-9]",
                           log or "", re.M))
        t.update(state="running", frac=min(n / 9.0, 1.0),
                 note=f"{n}/9 tensors done")
    elif log:
        t.update(state="failed", frac=0.0,
                 note="log exists but no artifact — check log")
    else:
        t.update(state="pending", frac=0.0, note="")
    tasks["t2"] = t

    # ---- Task 3: codebook gain on originals
    t = {"id": "t3", "label": TASKS[3][1], "weight": TASKS[3][2]}
    log = tail(os.path.join(FP16, "codebook_gain_fp16.log"))
    logs["t3"] = log
    skip3 = read_json(os.path.join(FP16, "task3_skip.json"))
    if cbg:
        t.update(state="done", frac=1.0, note="wrote fp16/codebook_gain.json")
    elif skip3 and skip3.get("skipped"):
        t.update(state="skipped", frac=1.0,
                 note="SKIPPED per plan gate: " + skip3.get("reason", ""))
    elif "t3" in st["procs"]:
        n = (log or "").count("[done]")
        t.update(state="running", frac=min(n / 9.0, 1.0),
                 note=f"{n}/9 tensors done")
    elif log:
        t.update(state="failed", frac=0.0,
                 note="log exists but no artifact — check log")
    else:
        t.update(state="pending", frac=0.0, note="")
    tasks["t3"] = t

    # ---- Task 4: report
    t = {"id": "t4", "label": TASKS[4][1], "weight": TASKS[4][2]}
    t.update(state="done" if st["findings_exists"] else "pending",
             frac=1.0 if st["findings_exists"] else 0.0,
             note="findings doc written" if st["findings_exists"] else "")
    tasks["t4"] = t

    st["tasks"] = tasks
    st["overall"] = sum(t["weight"] * t["frac"] for t in tasks.values())

    # ---- side info
    side = []
    if fetch:
        cfg = fetch.get("config", {})
        dtype = next((v.get("dtype_on_disk")
                      for v in fetch.get("tensor_check", {}).values()), "?")
        side += ["== Task 0: fetch ==",
                 f"  repo {fetch.get('repo')} @ {str(fetch.get('revision'))[:12]}",
                 f"  verdict {fetch.get('verdict')}  shards {fetch.get('shards_on_disk')}"
                 f"  {fetch.get('bytes_on_disk', 0) / 1e9:.1f} GB on disk",
                 f"  dtype on disk (safetensors headers): {dtype}",
                 f"  tie_word_embeddings: {cfg.get('tie_word_embeddings')}",
                 "  cross-check original vs dequant W4A16 (layer 27):"]
        for k, v in fetch.get("cross_check", {}).items():
            side.append(f"    {k.split('layers.27.')[-1]:28s} corr={v['corr']:.6f}"
                        f"  rel_mse={v['rel_mse_vs_orig_var']:.2e}")
        side.append(f"  disk free after: {fetch.get('disk_free_after_gb')} GB")
    if het:
        floor = max((v["worst_eta2"] for v in het.get("_iid_floor", {}).values()),
                    default=0.0)
        side += ["", "== Task 2: eta^2 originals vs baseline (best salt/tensor) ==",
                 f"  iid noise floor (worst over shapes): {floor:.2e}"]
        for k, v in het.items():
            if k.startswith("_"):
                continue
            b = v.get("best", {})
            bb = (het_base or {}).get(k, {}).get("best", {})
            flag = " <-- ABOVE FLOOR" if b.get("eta_squared", 0) > 10 * floor else ""
            side.append(f"  {k:38s} {b.get('salt', '?'):>10s}"
                        f" eta2={b.get('eta_squared', 0):.6f}"
                        f"   [base {bb.get('salt', '?'):>10s}"
                        f" {bb.get('eta_squared', 0):.6f}]{flag}")
    if cbg:
        side += ["", "== Task 3: codebook gain originals vs baseline (min r_s) =="]
        for k, v in cbg.items():
            if k.startswith("_"):
                continue
            rs = [v["bits"][b]["ratio_salted"] for b in ("2", "3", "4")]
            if cbg_base and k in cbg_base:
                rb = [cbg_base[k]["bits"][b]["ratio_salted"] for b in ("2", "3", "4")]
            else:
                rb = [float("nan")] * 3
            side.append(f"  {k:38s} min r_s orig={min(rs):.4f}"
                        f"  base={min(rb):.4f}"
                        f"  {'SALTED WINS' if min(rs) < 0.9 else 'no gain'}")
        hl = cbg.get("_spec_headline", {})
        if hl:
            side.append("  spec headline (salted 2-bit vs global 4-bit MSE):"
                        f" best orig={min(v['salted2_vs_global4_orig'] for v in hl.values()):.4f}"
                        f"  base={min(v['salted2_vs_global4_baseline'] for v in hl.values()):.4f}"
                        "  (~1.0 = spec claim)")
    st["side"] = side
    st["logs"] = logs
    st["active"] = next((t for t in tasks.values() if t["state"] == "running"),
                        None)
    return st


def render(widgets, st):
    widgets["overall_bar"]["value"] = st["overall"] * 100
    widgets["overall_var"].set(f"overall {st['overall'] * 100:.0f}%")
    for tid, _label, _w in TASKS:
        t = st["tasks"][tid]
        word, bg, fg = STATUS_STYLE[t["state"]]
        w = widgets["tasks"][tid]
        w["status"].config(text=word, bg=bg, fg=fg)
        w["bar"]["value"] = t["frac"] * 100
        w["note"].config(text=t["note"])
    act = st["active"]
    if act:
        p = st["procs"].get(act["id"], {})
        widgets["detail_var"].set(
            f"ACTIVE: {act['label']}   pid={p.get('pid', '?')}  "
            f"elapsed={p.get('elapsed_s', '?')}s   [{st['time']}]")
        log = st["logs"].get(act["id"]) or \
            "(no log file yet — process output is in the Hermes terminal)"
        widgets["task_bar"]["value"] = act["frac"] * 100
    else:
        widgets["detail_var"].set(f"no active probe process   [{st['time']}]")
        log = ""
        widgets["task_bar"]["value"] = 0
    widgets["log"].config(state="normal")
    widgets["log"].delete("1.0", "end")
    widgets["log"].insert("1.0", log)
    widgets["log"].see("end")
    widgets["log"].config(state="disabled")
    widgets["side"].config(state="normal")
    widgets["side"].delete("1.0", "end")
    widgets["side"].insert("1.0", "\n".join(st["side"]) or "(no artifacts yet)")
    widgets["side"].config(state="disabled")


def main():
    if "--once" in sys.argv:
        st = gather_state()
        st["logs"] = {k: (v or "")[-400:] for k, v in st["logs"].items()}
        print(json.dumps(st, indent=1, default=str))
        return

    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("salted-quantization FP16-originals probe")

    widgets = {}
    top = ttk.Frame(root, padding=6)
    top.grid(row=0, column=0, sticky="ew")
    widgets["overall_bar"] = ttk.Progressbar(top, length=560, maximum=100)
    widgets["overall_bar"].grid(row=0, column=0, padx=4)
    widgets["overall_var"] = tk.StringVar()
    ttk.Label(top, textvariable=widgets["overall_var"]).grid(row=0, column=1)

    tasks_frame = ttk.LabelFrame(root, text="tasks", padding=6)
    tasks_frame.grid(row=1, column=0, sticky="ew", padx=6, pady=4)
    widgets["tasks"] = {}
    for i, (tid, label, _w) in enumerate(TASKS):
        f = ttk.Frame(tasks_frame)
        f.grid(row=i, column=0, sticky="ew", pady=1)
        st_lbl = tk.Label(f, text="....", width=5, bg="#eeeeee",
                          font=("TkDefaultFont", 9, "bold"))
        st_lbl.grid(row=0, column=0, padx=2)
        ttk.Label(f, text=label).grid(row=0, column=1, sticky="w", padx=4)
        bar = ttk.Progressbar(f, length=180, maximum=100)
        bar.grid(row=0, column=2, padx=4)
        note = ttk.Label(f, text="", width=95, anchor="w")
        note.grid(row=0, column=3, sticky="w", padx=4)
        widgets["tasks"][tid] = {"status": st_lbl, "bar": bar, "note": note}

    detail = ttk.LabelFrame(root, text="active task log (tail)", padding=6)
    detail.grid(row=2, column=0, sticky="ew", padx=6, pady=4)
    widgets["detail_var"] = tk.StringVar()
    ttk.Label(detail, textvariable=widgets["detail_var"]).grid(
        row=0, column=0, sticky="w")
    widgets["task_bar"] = ttk.Progressbar(detail, length=560, maximum=100)
    widgets["task_bar"].grid(row=1, column=0, sticky="ew", pady=2)
    log = tk.Text(detail, height=12, width=110, font=("Courier", 9),
                  state="disabled")
    log.grid(row=2, column=0, sticky="ew")
    widgets["log"] = log

    side_frame = ttk.LabelFrame(root, text="results so far", padding=6)
    side_frame.grid(row=3, column=0, sticky="ew", padx=6, pady=4)
    side = tk.Text(side_frame, height=16, width=110, font=("Courier", 9),
                   state="disabled")
    side.grid(row=0, column=0, sticky="ew")
    widgets["side"] = side

    def refresh():
        try:
            render(widgets, gather_state())
        except Exception as e:  # a poll error must never kill the GUI
            widgets["detail_var"].set(f"poll error: {e}")
        root.after(POLL_MS, refresh)

    refresh()
    root.mainloop()


if __name__ == "__main__":
    main()
