import json

si_file = "flores-ted/valid.en_XX"
en_file = "flores-ted/valid.si_LK"

si_file = open(si_file).readlines()
en_file = open(en_file).readlines()

out = dict()
out["data"] = []
for i, j in zip(si_file, en_file):
    ins = dict()
    ins["si"] = i.strip()
    ins["en"] = j.strip()
    out["data"].append(ins)

output_file = open("flores-ted-6k-valid.json", "w")
json.dump(out, output_file, ensure_ascii=False, indent=4)
