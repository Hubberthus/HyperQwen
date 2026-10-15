import json
for f in ["smoke_ar.json", "smoke_baseline.json", "smoke_negctrl.json"]:
    d = json.load(open(f))
    pin = d.get("pin_report")
    if pin:
        print("== %s pins ==" % f)
        print(json.dumps(pin, indent=1)[:1500])
    rows = d.get("results", [])
    print("== %s: %d rows ==" % (f, len(rows)))
    for r in rows:
        toks = r.get("tokens") or [r.get("token")]
        print("  %-42s %s" % (r.get("prompt", "")[:40], "|".join(str(t) for t in toks[:6])))
