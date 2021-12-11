import json

source1 = "data/newsela-splitted-sineval/test.complex"
source2 = "data/newsela-splitted-sineval/test.simple"
source1_label = "com"
source2_label = "sim"
output_file = "test.json"

source1 = open(source1).readlines()
source2 = open(source2).readlines()

out_file = open(output_file, "w")
out = dict()
out["data"] = []
assert len(source1) == len(source2)
for i in range(len(source1)):
    ins = {source1_label: source1[i].strip(), source2_label: source2[i].strip()}
    out["data"].append(ins)
json.dump(out, out_file, ensure_ascii=False, indent=4)
