import json
from typing import List, Dict


# mined ==> po,pt
# newsela si ==> com,sim
# newsela ==> come,sime


def gen_data_from_dataset(data_file_name1: str, data_file_name2: str, prefix="po-pt: ", size=12):
    data1 = [prefix + line.strip() for line in open(data_file_name1).readlines()]
    data2 = [prefix + line.strip() for line in open(data_file_name2).readlines()]
    assert len(data1) == len(data2)
    start = 0
    for ins in range(int(len(data1) / size)):
        yield data1[start:start + size], data2[start:start + size]
        start = start + size


def create_new_dataset(key1_list: List[str], key2_list: List[str], new_key1="lang1", new_key2="lang2"):
    new_dataset = dict()
    new_dataset["data"] = []
    assert len(key1_list) == len(key2_list)
    for i, j in zip(key1_list, key2_list):
        new_dataset["data"].append({new_key1: i, new_key2: j})
    return new_dataset


def write_dataset_to_file(dataset: Dict, file_name="new_dataset.json"):
    file = open(file_name, "w")
    json.dump(dataset, file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    new_data_key1 = []
    new_data_key2 = []
    per_batch = 4
    min_data_size = 56603
    ep = min_data_size//per_batch
    gen_para = gen_data_from_dataset("../data/newsela-en/train.complex", "../data/newsela-en/train.simple",
                                     "come-sime: ", per_batch)
    gen_trans = gen_data_from_dataset("../data/sita56k/parallel-27.04.2021-trUnique56K.si-en-ta.en",
                                      "../data/sita56k/parallel-27.04.2021-trUnique56K.si-en-ta.si", "en-si: ", per_batch)
    for i in range(ep):
        d1, d2 = next(gen_para)
        new_data_key1.extend(d1)
        new_data_key2.extend(d2)

        d1, d2 = next(gen_trans)
        new_data_key1.extend(d1)
        new_data_key2.extend(d2)

    print(len(new_data_key1))

    # Add task token here if required
    data_dict = create_new_dataset(new_data_key1, new_data_key2, new_key1="lang1", new_key2="lang2")
    write_dataset_to_file(data_dict, "new/multitask-tr-sime.json")
