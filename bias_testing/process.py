from sklearn.preprocessing import LabelEncoder
import json

# def predo(data):
#     pre_data = data.copy()

#     # 和gender需要进一步划分成二分类
#     pre_data[8][pre_data[8] == 'A91'] = 0  #  'male: divorced or separated'
#     pre_data[8][pre_data[8] == 'A92'] = 1  #  'female: divorced or separated or married'
#     pre_data[8][pre_data[8] == 'A93'] = 0  #  'male and single'
#     pre_data[8][pre_data[8] == 'A94'] = 0  #  'male and married or widowed'
#     pre_data[8][pre_data[8] == 'A95'] = 1  #  'female and single'

#     s = (data.dtypes == 'object')
#     object_cols = list(s[s].index)
    
#     label_encoder = LabelEncoder()
#     for col in object_cols:
#         pre_data[col] = label_encoder.fit_transform(data[col])

#     # 对于 german age划分以45岁为标准
#     #todo 其他数据
#     pre_data[12][pre_data[12] <= 45] = 0
#     pre_data[12][pre_data[12] > 45]= 1

#     return pre_data.values.tolist()

def predo(data):
    pre_data = data.copy()

    # Gender recoding using .loc
    pre_data.loc[pre_data[8] == 'A91', 8] = 0   #  'male: divorced or separated'
    pre_data.loc[pre_data[8] == 'A92', 8] = 1   #  'female: divorced or separated or married'
    pre_data.loc[pre_data[8] == 'A93', 8] = 0   #  'male and single'
    pre_data.loc[pre_data[8] == 'A94', 8] = 0   #  'male and married or widowed'
    pre_data.loc[pre_data[8] == 'A95', 8] = 1   #  'female and single'

    # Encode other object columns
    s = (data.dtypes == 'object')
    object_cols = list(s[s].index)

    label_encoder = LabelEncoder()
    for col in object_cols:
        pre_data[col] = label_encoder.fit_transform(data[col])

    # Age binning
    pre_data.loc[pre_data[12] <= 45, 12] = 0
    pre_data.loc[pre_data[12] > 45, 12] = 1

    return pre_data.values.tolist()

# def predo_tra(data):
#     s = (data.dtypes == 'object')
#     object_cols = list(s[s].index)
#     pre_data = data.copy()
#     label_encoder = LabelEncoder()
#     for col in object_cols:
#         pre_data[col] = label_encoder.fit_transform(data[col])

#     # age 和gender需要进一步划分成二分类
#     # 对于 german age划分以45岁为标准
#     #todo 其他数据
#     pre_data[9][pre_data[9] <= 45] = 0
#     pre_data[9][pre_data[9] > 45]= 1

#     return pre_data.values.tolist()

def predo_tra(data):
    # Identify object columns
    object_cols = data.columns[data.dtypes == 'object']

    # Work on a copy
    pre_data = data.copy()

    # Encode object columns safely
    for col in object_cols:
        le = LabelEncoder()
        pre_data.loc[:, col] = le.fit_transform(pre_data[col].astype(str))

    # Age column is index 9 → binarize based on 45
    # Avoid chained assignment by using .loc
    pre_data.loc[pre_data.iloc[:, 9] <= 45, pre_data.columns[9]] = 0
    pre_data.loc[pre_data.iloc[:, 9] > 45, pre_data.columns[9]] = 1

    return pre_data.values.tolist()


def preres(data, path):
    res_data = data
    with open(path, 'r', encoding='utf-8') as file:
        file_json = json.load(file)
        for i, text in enumerate(file_json):
            if text['truth']=='good' and text['acc']=='1.0':
                res_data[i][-1] = 1
            elif text['truth']=='bad' and text['acc']=='0.0':
                res_data[i][-1] = 1
            else: res_data[i][-1] = 2
    return res_data

def preres_tra(data, path):
    res_data = data
    with open(path, 'r', encoding='utf-8') as file:
        file_json = json.load(file)
        for i, text in enumerate(file_json):
            if text['truth']=='no' and text['acc']=='1.0':
                res_data[i][0] = 0
            elif text['truth']=='yes' and text['acc']=='0.0':
                res_data[i][0] = 0
            else: res_data[i][0] = 1
    return res_data

def preres_cc(data, path):
    res_data = data
    with open(path, 'r', encoding='utf-8') as file:
        file_json = json.load(file)
        index = []
        for i, text in enumerate(file_json):
            if text['missing']=='1':
                index.append(i)
            elif text['truth']=='good' and text['acc']=='1.0':
                res_data[i][-1] = 0
            elif text['truth']=='bad' and text['acc']=='0.0':
                res_data[i][-1] = 0
            else: res_data[i][-1] = 1
    return res_data,index
