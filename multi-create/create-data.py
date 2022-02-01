import json
import random
from typing import List, Dict


# mined ==> po,pt
# newsela si ==> com,sim
# newsela ==> come,sime


def get_data_from_dataset(data_file_name: str, key1="com", key2="sim"):
    key1_list = []
    key2_list = []
    dataset = json.load(open(data_file_name))['data']
    for ins in dataset:
        key1_list.append(ins[key1])
        key2_list.append(ins[key2])
    assert len(key1_list) == len(key2_list)
    return key1_list, key2_list


def create_new_dataset(key1_list: List[str], key2_list: List[str], new_key1="lang1", new_key2="lang2"):
    new_dataset = dict()
    new_dataset["data"] = []
    assert len(key1_list) == len(key2_list)
    for i, j in zip(key1_list, key2_list):
        new_dataset["data"].append({new_key1: i, new_key2: j})
    random.shuffle(new_dataset["data"])
    return new_dataset


def write_dataset_to_file(dataset: Dict, file_name="new_dataset.json"):
    file = open(file_name, "w")
    json.dump(dataset, file, ensure_ascii=False, indent=4)


def do_subsampling(key1_list: List, key2_list: List, limit=7000, randomize=False):
    if randomize:
        zipped = list(zip(key1_list, key2_list))
        random.shuffle(zipped)
        key1_list, key2_list = zip(*zipped)
        # todo check randomize
    return key1_list[:limit], key2_list[:limit]


if __name__ == '__main__':
    file_names = ["mined-7k-valid-tt.json", "newsela-en-valid-tt.json", "newsela-si-valid-tt.json",
                  "sita-56k-valid-tt.json"]
    keys = ["po,pt", "come,sime", "com,sim", "en,si"]

    new_data_key1 = []
    new_data_key2 = []
    for name, keys in zip(file_names, keys):
        k1 = keys.split(",")[0].strip()
        k2 = keys.split(",")[1].strip()
        d1, d2 = get_data_from_dataset(name, k1, k2)
        # Do subsampling here if required
        d1, d2 = do_subsampling(d1, d2, limit=7000, randomize=True)

        new_data_key1.extend(d1)
        new_data_key2.extend(d2)
    print(len(new_data_key1))

    # Add task token here if required
    data_dict = create_new_dataset(new_data_key1, new_data_key2, new_key1="lang1", new_key2="lang2")
    write_dataset_to_file(data_dict, "new/multitask-7000-valid.json")
