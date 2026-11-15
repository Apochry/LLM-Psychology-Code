import re, json
from collections import Counter
from datasets import load_dataset
def show(name, split="train", **kw):
    ds = load_dataset(name, split=split, **kw)
    print("==", name, len(ds), ds.column_names)
    return ds
d = show("allenai/olmo-2-hard-coded")
for r in d.select(range(4)):
    print(json.dumps(r, ensure_ascii=False)[:600])
d2 = show("allenai/tulu-3-hard-coded-10x")
print(json.dumps(d2[0], ensure_ascii=False)[:500])
# Does the OLMo-2 SFT mixture include them? count by source
m = show("allenai/tulu-3-sft-olmo-2-mixture")
c = Counter(m["source"]); print(c.most_common(30))
