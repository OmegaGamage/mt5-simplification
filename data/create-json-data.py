import json

si_file = "sita56k/parallel-27.04.2021-trUnique56K.si-en-ta.si"
en_file = "sita56k/parallel-27.04.2021-trUnique56K.si-en-ta.en"
ta_file = "sita56k/parallel-27.04.2021-trUnique56K.si-en-ta.ta"
si_file = open(si_file).readlines()
en_file = open(en_file).readlines()
ta_file = open(ta_file).readlines()

out = dict()
out["data"] = []
for i, j, k in zip(si_file, en_file, ta_file):
    ins = dict()
    ins["si"] = i.strip()
    ins["en"] = j.strip()
    ins["ta"] = k.strip()
    out["data"].append(ins)

output_file = open("sita56k.json", "w")
json.dump(out, output_file, ensure_ascii=False, indent=4)
