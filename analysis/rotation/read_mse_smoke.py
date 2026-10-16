import json
d = json.load(open("/home/hubbe/qwen-serving/analysis/rotation/smoke_mse.json"))
for r in d["results"][:6]:
    print(repr(r["text"]))
pr = d["pin_report"]["quant_method_classes"]
from collections import Counter
print(Counter(pr.values()))
