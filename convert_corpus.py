import pandas as pd
from nltk.corpus import treebank

tm = {
    '``': "'",
    "''": "'",
    '-LRB-': "(",
    '-RRB-': ")",
    '--': "–",
}

lines = []

for f in treebank.fileids():
    for s in treebank.sents(f):
        out = ""
        for i, t in enumerate(s):
            mt = t
            if "*" in t or t == "0":
                continue
            if t in [",", ".", "n't", "'re", "'ve", "'s", "''", "%", "-RRB-", "'", ";", ":"]:
                out = out[:-1]
            if t in tm:
                mt = tm[t]
            out += mt
            if i + 1 < len(s) and t not in ["$", "``", "-LRB-"]:
                out += " "
#        print(f"{i} >>{out}<<")
        lines.append(out)

df = pd.DataFrame(lines, columns=['line'])
df.to_parquet("lines.parquet")
df.to_csv("lines.csv", index=False)
