from sklearn.preprocessing import LabelEncoder
import json
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

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

    # Gender recoding using .replace() — NOTE: pandas 3.0's 'string' dtype rejects
    # mixing int values into a still-string column via element-wise .loc assignment
    # (raises TypeError: Invalid value '0' for dtype 'str'). .replace() correctly
    # converts the column's dtype instead.
    pre_data[8] = pre_data[8].replace({
        'A91': 0,  # male: divorced or separated
        'A92': 1,  # female: divorced or separated or married
        'A93': 0,  # male and single
        'A94': 0,  # male and married or widowed
        'A95': 1,  # female and single
    })

    # Encode remaining object/string columns — EXCLUDE column 8, it's already recoded above
    # NOTE: pandas 3.0+ infers CSV text columns as the new 'string' dtype rather than
    # legacy 'object', so we must check for both or these columns get silently skipped.
    object_cols = data.select_dtypes(include=['object', 'string']).columns.tolist()
    object_cols = [col for col in object_cols if col != 8]

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
    # Identify object/string columns
    # NOTE: pandas 3.0+ infers CSV text columns as the new 'string' dtype rather than
    # legacy 'object', so we must check for both or these columns get silently skipped.
    object_cols = data.select_dtypes(include=['object', 'string']).columns.tolist()
    pre_data = data.copy()
    label_encoder = LabelEncoder()
    for col in object_cols:
        pre_data[col] = label_encoder.fit_transform(data[col])

    # Age column is index 9 → binarize based on 45
    # Avoid chained assignment by using .loc
    pre_data.loc[pre_data.iloc[:, 9] <= 45, pre_data.columns[9]] = 0
    pre_data.loc[pre_data.iloc[:, 9] > 45, pre_data.columns[9]] = 1

    return pre_data.values.tolist()


def preres(data, path):
    res_data = data
    index = []
    with open(path, 'r', encoding='utf-8') as file:
        file_json = json.load(file)
        for i, text in enumerate(file_json):
            if text['missing'] == '1':
                index.append(i)
            elif text['truth']=='good' and text['acc']=='1.0':
                res_data[i][-1] = 1
            elif text['truth']=='bad' and text['acc']=='0.0':
                res_data[i][-1] = 1
            else: res_data[i][-1] = 2
    return res_data, index

def preres_tra(data, path, target_index = 4):
    res_data = data
    index = []
    # When model predicts "no" for a record, we set target_index value to 0.
    with open(path, 'r', encoding='utf-8') as file:
        file_json = json.load(file)
        for i, text in enumerate(file_json):
            if text['missing'] == '1':
                index.append(i)
            elif text['truth']=='no' and text['acc']=='1.0':
                res_data[i][target_index] = 0
            elif text['truth']=='yes' and text['acc']=='0.0':
                res_data[i][target_index] = 0
            else: res_data[i][target_index] = 1
    return res_data, index

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

def load_json_list(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON file must contain a list of objects.")

    return data

def compute_metrics(output_filepath, positive_choice = 'good'):
    records = load_json_list(output_filepath)
    # Extract true and predicted labels
    y_true = [1 if r['truth'].strip().lower() == positive_choice else 0 for r in records]
    y_pred = [1 if r['logit_0'].strip().lower() == positive_choice else 0 for r in records]

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    return {'accuracy': acc, 'F1-score': f1, 'MCC': mcc}