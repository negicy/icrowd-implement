import pandas as pd
import numpy as np
import pickle

label_df = pd.read_csv("label_df.csv", sep = ",")
batch_df = pd.read_csv("batch_100.csv", sep = ",")
label_df = label_df.set_index('id')
input_df = pd.read_csv("input.csv", sep = ",")

# origin_id: max(prob_dic)のdict作成
task_dic = {}
for k in range(0, 100):
  task_id = "q"+str(k+1)
  # 元のテストデータのIDを取り出す
  origin_index = 'Input.'+task_id
  origin_id = batch_df[origin_index][0]
  task_dic["q"+str(k+1)] = origin_id

# print(task_dic)
input_df = input_df.set_index('qid')
input_df['task_id'] = 0

q_list = list(input_df.index)
# print(q_list)

# Task IDリストの作成
task_list = list(task_dic.values())
# print(task_list)

# input_dfのインデックスを置き換え
for q in q_list:
  input_df['task_id'][q] = task_dic[q]
input_df = input_df.set_index('task_id')

worker_list = list(input_df.columns)

main_objects = {}
main_objects["worker_list"] = worker_list
main_objects["task_list"] = task_list
main_objects["input_df"] = input_df


# worker_list, task_list, 
with open("main_objects.pickle", mode="wb") as f:
    pickle.dump(main_objects, f)
  