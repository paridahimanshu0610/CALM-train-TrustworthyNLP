import pandas as pd
import numpy as np
import sklearn as sk
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.datasets import BinaryLabelDataset
from aif360.explainers import MetricTextExplainer
import random
import json
from process import predo, preres_cc, compute_metrics
import os

'''data preprocess'''
feature_size = 8


# 每个数据的变量名
mean_list = ['gender', 'state','cardholder','balance','numTrans',
             'numIntTrans','creditLine','fraudRisk']


# 原数据处理
# data中所有数据需要修改成数值格式
# todo age 和gender需要进一步划分成二分类？
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def prepare_input_data(filename, output_file = None):
    my_data = pd.read_csv(filename, sep=',', names=[i for i in range(feature_size)])
    my_data_df = pd.DataFrame(my_data)
    my_data_df.columns = mean_list
    if output_file:
        _, index_to_drop = preres_cc(my_data_df.values.tolist(), output_file)
        my_data_df = my_data_df.drop(index_to_drop)      
    return my_data_df

def prepare_output_data(output_filename, test_filename):
    my_test_df = prepare_input_data(test_filename)
    my_data, my_idx = preres_cc(my_test_df.values.tolist(), output_filename)
    my_data_df = pd.DataFrame(my_data)
    my_data_df.columns = mean_list
    if len(my_idx) > 0:
        my_data_df = my_data_df.drop(my_idx)
    return my_data_df

def disparate_impact(input_df):
    input_data = BinaryLabelDataset(favorable_label=0, unfavorable_label=1, df=input_df, label_names=['fraudRisk'], protected_attribute_names=['gender'])
    final_res = dict()
    
    # Gender DI
    metric = BinaryLabelDatasetMetric(input_data, unprivileged_groups=[{'gender':2}], privileged_groups=[{'gender':1}])
    # text_res = MetricTextExplainer(metric)        
    final_res['Gender'] = metric.disparate_impact()

    return final_res

def bias_test(output_df, input_test_df):
    llm_output_data = BinaryLabelDataset(favorable_label=0, unfavorable_label=1, df=output_df, label_names=['fraudRisk'], protected_attribute_names=['gender'])
    input_test_data = BinaryLabelDataset(favorable_label=0, unfavorable_label=1, df=input_test_df, label_names=['fraudRisk'], protected_attribute_names=['gender'])
    final_res = {'EOD': {}, "AOD": {}}
    
    # Gender EOD and AOD
    metric = ClassificationMetric(input_test_data, llm_output_data, unprivileged_groups=[{'gender':2}], privileged_groups=[{'gender':1}])
    # text_res = MetricTextExplainer(metric)        
    final_res['EOD']["Gender"] = metric.equal_opportunity_difference()
    final_res['AOD']["Gender"] = metric.average_odds_difference()

    return final_res

model_name = "CALM"
prompt_file_suffix = "_cf" # "_zero_shot" | "_cf"

train_filename = os.path.join(project_dir, "data", "split_data", "ccFraud_fraud_detection", "bias_data", "ccfraud_train.csv")
all_test_filename = os.path.join(project_dir, "data", "split_data", "ccFraud_fraud_detection", "bias_data", "ccfraud_test.csv")
test_filename = os.path.join(project_dir, "data", "split_data", "ccFraud_fraud_detection", "bias_data", "ccFraud_gender_split.csv")
output_filename = os.path.join(project_dir, "inference", "model_inference", model_name, "ccFraud_fraud_detection", "ccfraud_gender" + prompt_file_suffix + ".json")

train = prepare_input_data(train_filename)
all_test =prepare_input_data(all_test_filename)
test = prepare_input_data(test_filename, output_file=output_filename)
res = prepare_output_data(output_filename, test_filename)

print("Train DI:", disparate_impact(train))
print("Train DI:", disparate_impact(all_test))
print("Bias Test:", bias_test(res, test))
print("Results:", compute_metrics(output_filename, positive_choice='good'))