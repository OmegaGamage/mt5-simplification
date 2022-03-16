import itertools

import sklearn
import statsmodels
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa

lines = 25


def get_model_pd(file_path: str, model_name: str, column_name: str) -> list:
    data = pd.ExcelFile(file_path)
    df = pd.read_excel(data, model_name, convert_float=False)[:lines].fillna(0)
    sr = df[column_name]
    return sr.tolist()


def get_cohens_kappa(p, q) -> float:
    kappa = round(cohen_kappa_score(p, q), 2)
    return kappa


# def get_fleiss_iaa(p, q, r) -> float:
#     new_arr = []
#     new_arr[0] = new_arr[p.tolist(), q.tolist(), r.tolist()]
#     new_arr[1] = ['p', 'q', 'r']
#     # new_arr['p'] = p
#     # new_arr['q'] = q
#     # new_arr['r'] = r
#     k = fleiss_kappa(new_arr)
#     return k


cols = ['Hallucinations', 'Fluency Errors', 'Anaphora Res.',
        'Bad substitution', 'Near Exact Copy', 'Missing Major Fact']
file_names = ["data/Thamindu.xlsx", "data/Lahiru.xlsx", "data/Rumesh.xlsx"]
model_names = ["mBART-model1", "mBART-model2", "mBART-model8", "mBART-model18"]


def iaa_for_error_type(column_name):
    single_person = []
    for annot in file_names:
        lst = []
        for model in model_names:
            lst.extend(get_model_pd(annot, model, column_name))
        single_person.append(lst)
    comb = itertools.combinations(single_person, 2)
    # print(list(comb))
    out = []
    for com in comb:
        out.append(get_cohens_kappa(com[0], com[1]))
    return out


# column = cols[1]
# a = get_model_pd(file_names[0], model_names[0], column)
# b = get_model_pd(file_names[1], model_names[0], column)
# c = get_model_pd(file_names[2], model_names[0], column)
# print(a)
# print("pairwise IAA for ", cols[4], model_names[0])
# print(get_cohens_kappa(a, b), get_cohens_kappa(c, b), get_cohens_kappa(a, c))
print("Pairwise Cohens Kappa")
for err in cols:
    print(err)
    print(iaa_for_error_type(err))
    print("=============")
# print(iaa_for_error_type("Near Exact Copy"))
