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


mean_list = ['target','Agency','Agency Type','Distribution Channel','Product Name',
             'Duration','Destination','Net Sales','Commission','Age']
feature_size = 10


current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)


def prepare_input_data(filename):
    input_data = pd.read_csv(filename, sep=',', names=[i for i in range(feature_size)])
    input_data = predo_tra(input_data)
    input_data_df = pd.DataFrame(input_data)
    input_data_df.columns = mean_list
    return input_data_df   

def prepare_output_data(output_filename, test_data_df):
    output_data = preres_tra(test_data_df.values.tolist(), output_filename)
    output_data_df = pd.DataFrame(output_data)
    output_data_df.columns = mean_list
    return output_data_df    

def disparate_impact(input_df):
    input_data = BinaryLabelDataset(favorable_label=0, unfavorable_label=1, df=input_df, label_names=['target'], protected_attribute_names=['Age'])
    final_res = dict()
    
    # Gender DI
    metric = BinaryLabelDatasetMetric(input_data, unprivileged_groups=[{'Age':1}], privileged_groups=[{'Age':0}])
    # text_res = MetricTextExplainer(metric)        
    final_res['Age'] = metric.disparate_impact()

    return final_res

def bias_test(output_df, input_test_df):
    llm_output_data = BinaryLabelDataset(favorable_label=0, unfavorable_label=1, df=output_df, label_names=['target'], protected_attribute_names=['Age'])
    input_test_data = BinaryLabelDataset(favorable_label=0, unfavorable_label=1, df=input_test_df, label_names=['target'], protected_attribute_names=['Age'])
    final_res = {'EOD': {}, "AOD": {}}
    
    # Gender EOD and AOD
    metric = ClassificationMetric(input_test_data, llm_output_data, unprivileged_groups=[{'Age':1}], privileged_groups=[{'Age':0}])
    # text_res = MetricTextExplainer(metric)        
    final_res['EOD']["Age"] = metric.equal_opportunity_difference()
    final_res['AOD']["Age"] = metric.average_odds_difference()

    return final_res

train_filename = os.path.join(current_dir, 'bias_data', 'TraIn_train.csv')
test_filename = os.path.join(current_dir, 'bias_data', 'TraIn_test.csv')
output_filename = os.path.join(current_dir, 'CALM', 'flare_trin_desc_write_out_info.json')

train = prepare_input_data(train_filename)
test = prepare_input_data(test_filename)
res = prepare_output_data(output_filename, test)

print("Train DI:", disparate_impact(train))
print("Test DI:", disparate_impact(test))
print("Bias Test:", bias_test(res, test))
print("Results:", compute_metrics(output_filename, positive_choice='yes'))