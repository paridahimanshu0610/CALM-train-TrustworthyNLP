import pandas as pd
import numpy as np
import sklearn as sk
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.datasets import BinaryLabelDataset
from aif360.explainers import MetricTextExplainer
import random
import json
import os
from process import predo, preres, compute_metrics

'''data preprocess'''
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
feature_size = 20+1
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

mean_list = ['Status of existing checking account', 'Duration in month', 'Credit history', 'Purpose',
             'Credit amount', 'Savings account/bonds', 'Present employment since',
             'Installment rate in percentage of disposable income', 'Personal status and sex',
             'Other debtors / guarantors', 'Present residence since', 'Property', 'Age in years',
             'Other installment plans', 'Housing', 'Number of existing credits at this bank' ,'Job',
             'Number of people being liable to provide maintenance for' , 'Telephone' , 'foreign worker',
             'target']

def prepare_input_data(filename):
    my_data = pd.read_csv(filename, sep=',', names=[i for i in range(feature_size)])
    my_data = predo(my_data)
    my_data_df = pd.DataFrame(my_data)
    my_data_df.columns = mean_list
    return my_data_df

def prepare_output_data(output_filename, my_test_df):
    my_data = preres(my_test_df.values.tolist(), output_filename)
    my_data_df = pd.DataFrame(my_data)
    my_data_df.columns = mean_list
    return my_data_df

def disparate_impact(input_df, bias_attributes = ['Personal status and sex', 'Age in years','foreign worker'], desired_attribute = None):
    input_data = BinaryLabelDataset(favorable_label=1, unfavorable_label=2, df=input_df, label_names=['target'], protected_attribute_names=bias_attributes)
    final_res = dict()
    
    # Foreigner DI
    if (desired_attribute == "foreign") or (not desired_attribute):
        metric = BinaryLabelDatasetMetric(input_data, unprivileged_groups=[{'foreign worker':0}], privileged_groups=[{'foreign worker':1}])
        # text_res = MetricTextExplainer(metric)        
        final_res['Foreigner'] = metric.disparate_impact()

    # Age DI
    if (desired_attribute == "age") or (not desired_attribute):
        metric = BinaryLabelDatasetMetric(input_data, unprivileged_groups=[{'Age in years':1}], privileged_groups=[{'Age in years':0}])
        # text_res = MetricTextExplainer(metric)        
        final_res['Age'] = metric.disparate_impact()

    # Gender DI
    if (desired_attribute == "gender") or (not desired_attribute):
        metric = BinaryLabelDatasetMetric(input_data, unprivileged_groups=[{'Personal status and sex':1}], privileged_groups=[{'Personal status and sex':0}])
        # text_res = MetricTextExplainer(metric)        
        final_res['Gender'] = metric.disparate_impact()

    return final_res

def bias_test(output_df, input_test_df, bias_attributes = ['Personal status and sex', 'Age in years','foreign worker'], desired_attribute = None):
    llm_output_data = BinaryLabelDataset(favorable_label=1, unfavorable_label=2, df=output_df, label_names=['target'], protected_attribute_names=bias_attributes)
    input_test_data = BinaryLabelDataset(favorable_label=1, unfavorable_label=2, df=input_test_df, label_names=['target'], protected_attribute_names=bias_attributes)
    final_res = {'EOD': {}, "AOD": {}}
    
    # Foreigner EOD and AOD
    if (desired_attribute == "foreign") or (not desired_attribute):
        metric = ClassificationMetric(input_test_data, llm_output_data, unprivileged_groups=[{'foreign worker':0}], privileged_groups=[{'foreign worker':1}])
        # text_res = MetricTextExplainer(metric)        
        final_res['EOD']["Foreigner"] = metric.equal_opportunity_difference()
        final_res['AOD']["Foreigner"] = metric.average_odds_difference()

    # Age EOD and AOD
    if (desired_attribute == "age") or (not desired_attribute):
        metric = ClassificationMetric(input_test_data, llm_output_data, unprivileged_groups=[{'Age in years':1}], privileged_groups=[{'Age in years':0}])
        # text_res = MetricTextExplainer(metric)      
        final_res['EOD']["Age"] = metric.equal_opportunity_difference()
        final_res['AOD']["Age"] = metric.average_odds_difference()

    # Gender EOD and AOD
    if (desired_attribute == "gender") or (not desired_attribute):
        metric = ClassificationMetric(input_test_data, llm_output_data, unprivileged_groups=[{'Personal status and sex':1}], privileged_groups=[{'Personal status and sex':0}])
        # text_res = MetricTextExplainer(metric)        
        final_res['EOD']["Gender"] = metric.equal_opportunity_difference()
        final_res['AOD']["Gender"] = metric.average_odds_difference()

    return final_res

model_name = "CALM"
current_target_feature = "foreign" # {"gender", "age", "foreign"}
prompt_file_suffix = "_zero_shot" # "_zero_shot"

train_filename = os.path.join(project_dir, "data", "split_data", "German_credit_scoring", "bias_data", "german_train.csv")
all_test_filename = os.path.join(project_dir, "data", "split_data", "German_credit_scoring", "bias_data", "german_test.csv")
test_filename = os.path.join(project_dir, "data", "split_data", "German_credit_scoring", "bias_data", "german_" + current_target_feature + "_split.csv")
output_filename = os.path.join(project_dir, "inference", "model_inference", model_name, "German_credit_scoring", "german_" + current_target_feature + prompt_file_suffix + ".json")

train = prepare_input_data(train_filename)
all_test =prepare_input_data(all_test_filename)
test = prepare_input_data(test_filename)
res = prepare_output_data(output_filename, test)

print("Train DI", disparate_impact(train, desired_attribute = current_target_feature))
print("Test DI:", disparate_impact(all_test, desired_attribute = current_target_feature))
print("Bias Test:", bias_test(res, test, desired_attribute = current_target_feature))
print("Results:", compute_metrics(output_filename, positive_choice='good'))