import pandas as pd
import numpy as np
import sklearn as sk
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.datasets import BinaryLabelDataset
from aif360.explainers import MetricTextExplainer
import random
import json
from process import predo_tra, preres_tra, compute_metrics
import os 


mean_list = ['Agency','Agency Type','Distribution Channel','Product Name','target',
             'Duration','Destination','Net Sales','Commission','Age']
feature_size = 10


current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def prepare_input_data(filename):
    input_data = pd.read_csv(filename, sep=',', names=[i for i in range(feature_size)])
    input_data = predo_tra(input_data)
    input_data_df = pd.DataFrame(input_data)
    input_data_df.columns = mean_list
    return input_data_df   

def prepare_output_data(output_filename, test_data_df):
    # When model predicts "no" for a record, we set target_index value to 0.
    output_data, index_to_drop = preres_tra(test_data_df.values.tolist(), output_filename, target_index=4)
    output_data_df = pd.DataFrame(output_data)
    output_data_df.columns = mean_list
    if index_to_drop:
        output_data_df = output_data_df.drop(index_to_drop).reset_index(drop=True)
    return output_data_df, index_to_drop

def disparate_impact(input_df):
    input_data = BinaryLabelDataset(favorable_label=0, unfavorable_label=1, df=input_df, label_names=['target'], protected_attribute_names=['Age'])
    final_res = dict()

    # Age DI
    metric = BinaryLabelDatasetMetric(input_data, unprivileged_groups=[{'Age':1}], privileged_groups=[{'Age':0}])
    final_res['Age'] = metric.disparate_impact()

    return final_res

def bias_test(output_df, input_test_df):
    # When model predicts "no" for a record, we set target_index value to 0. So, a model response of no (i.e. no claim filed) is considered as favorable (0).
    # Favorable label is the one which is considered as the positive outcome. In this case, a model response of no (i.e. no claim filed) is considered as a postive outcome.
    llm_output_data = BinaryLabelDataset(favorable_label=0, unfavorable_label=1, df=output_df, label_names=['target'], protected_attribute_names=['Age'])
    input_test_data = BinaryLabelDataset(favorable_label=0, unfavorable_label=1, df=input_test_df, label_names=['target'], protected_attribute_names=['Age'])
    final_res = {'EOD': {}, "AOD": {}, "AAOD": {}, "FPRD": {}, "ERR": {}, "ERD": {}}

    # Age EOD and AOD
    metric = ClassificationMetric(input_test_data, llm_output_data, unprivileged_groups=[{'Age':1}], privileged_groups=[{'Age':0}])
    final_res['EOD']["Age"] = metric.equal_opportunity_difference()
    final_res['AOD']["Age"] = metric.average_odds_difference()
    final_res['AAOD']["Age"] = metric.average_abs_odds_difference()
    final_res['FPRD']["Age"] = metric.false_positive_rate_difference()
    final_res['ERR']["Age"] = metric.error_rate_ratio()
    final_res['ERD']["Age"] = metric.error_rate_difference()

    return final_res

model_name = "CRA-llama3.1-8b-instruct_CRA_0.045M_checkpoint-7010"
prompt_file_suffix = "_bias" # "_zero_shot" | "_cf"

train_filename = os.path.join(project_dir, "data", "split_data", "Travel_Insurance", "bias_data", "TraIn_train.csv")
all_test_filename = os.path.join(project_dir, "data", "split_data", "Travel_Insurance", "bias_data", "TraIn_test.csv")
attribute_test_filename = os.path.join(project_dir, "data", "split_data", "Travel_Insurance", "bias_data", "travel_insurance_age_split.csv")
output_filename = os.path.join(project_dir, "inference", "hprc", "model_inference", model_name, "Travel_Insurance", "travel_insurance_age" + prompt_file_suffix + ".json")


train = prepare_input_data(train_filename)
test = prepare_input_data(all_test_filename)
attribute_test = prepare_input_data(attribute_test_filename)
res, dropped_idx = prepare_output_data(output_filename, attribute_test)

# Keep attribute_test aligned with res by dropping the same missing-response rows
if dropped_idx:
    attribute_test = attribute_test.drop(dropped_idx).reset_index(drop=True)

print("Train DI:", disparate_impact(train))
print("Test DI:", disparate_impact(test))
print("Bias Test:", bias_test(res, attribute_test))
# When model predicts "no" for a record, we set target_index value to 0. So, a model response of no (i.e. no claim filed) is considered as favorable (0).
# Favorable label is the one which is considered as the positive outcome (See doc of aif360.datasets.BinaryLabelDataset). In this case, a model response of no (i.e. no claim filed) is considered as a postive outcome.
print("Results:", compute_metrics(output_filename, positive_choice='no'))