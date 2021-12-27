import numpy as np
import pandas as pd
import sys, os
from icrowd_method import *
from candidate import *
from assignment_method import *
import matplotlib.pyplot as plt
import pickle

# Input Data
label_df = pd.read_csv("label_df.csv", sep = ",")
label_df = label_df.rename(columns={'Unnamed: 0': 'id'})
batch_df = pd.read_csv("batch_100.csv", sep = ",")
label_df = label_df.set_index('id')
input_df = pd.read_csv("input.csv", sep = ",")

d = {}
with open("main_objects.pickle", mode="rb") as f:
    d = pickle.load(f)

worker_list = d['worker_list']
task_list = d['task_list']
input_df = d['input_df']

# generate similarity graph
s = generate_similarity_graph(label_df, task_list)
print(s)
norm_sim_matrix = normalize_sim(s)
# print(norm_sim_matrix)


# run pagerank formula
iter_time = 100
p_vector = []


for t in range(0, len(task_list)):
    # p の初期値
    p_t = np.zeros(len(task_list))
    p_t[t] = 1
    q = p_t
   
    for i in range(0, iter_time):
        p_t = page_rank(norm_sim_matrix, p_t, q, a=0.5)
      
    p_vector.append(p_t)
    # print(p_vector)

# print(p_vector_list)
# print(p_vector_list)
'''
p_w_list = []

q_task = task_list[:60]
for worker in worker_list:
    q_w = np.zeros(len(task_list))
    for i in range(0, len(task_list)):
        task_id = task_list[i]
        if task_id in q_task:
            q_w[i] = input_df[worker][task_id]
    
    p_w = 0
    # p_t: vector, q: schalar
    for i in range(0, len(p_vector)):
        p_w += q_w[i]*p_vector[i]
        #p_w.append(p_t)
    # print(len(p_t))
    p_w_list.append(p_w)

print((p_w_list[0]))
'''

# 割り当てシミュレーション
# for i in range(0, iter_time):
acc_all_th = []
var_all_th = []

threshold = list([i / 100 for i in range(50, 81)])

# Solve for parameters
iteration_time = 15
for iteration in range(0, iteration_time):
  
  acc_per_th = []
  var_per_th = []

  results = make_candidate(threshold, input_df, p_vector, worker_list, task_list)

  worker_c_th = results[0]
  test_worker_set = results[1]

  # worker_c_th = {th: {task: [workers]}}
  # 各thresholdについて繰り返し
  for candidate_dic in worker_c_th.values():
    # print(candidate_dic)
    assign_dic = assignment(candidate_dic, test_worker_set)
    # 割り当て結果の精度を求める
    # print(assign_dic)
    acc = accuracy(assign_dic, input_df)
    var = task_variance(assign_dic, test_worker_set)

    acc_per_th.append(acc)
    var_per_th.append(var)

  acc_all_th.append(acc_per_th) 
  var_all_th.append(var_per_th)

mean_acc = [0] * len(threshold)
mean_var = [0] * len(threshold)
# 各スレッショルドについて, 全イテレーション間の平均を求める
for i in range(0, len(threshold)):
  nonecount = 0
  acc_sum = 0
  for acc_per_th in acc_all_th:
    if acc_per_th[i] == "NoneAssignment":
      nonecount += 1
    else:
      acc_sum += acc_per_th[i]
  mean_acc[i] = acc_sum / (iteration_time - nonecount)


'''
for acc_per_th in acc_all_th:
  # acc_sum = 0
  for i in range(0, len(threshold)):
    if acc_per_th[i] == "NoneAssignment":
      nonecount += 1
    else:
      mean_acc[i] += acc_list[i]
  mean_acc[i] = mean_acc[i] / iteration_time

  for var_list in var_all_th:
  # var_sum = 0
  for i in range(0, len(threshold)):
    mean_var[i] += var_list[i]


for i in range(0, len(threshold)):
  mean_acc[i] = mean_acc[i] / iteration_time
  mean_var[i] = mean_var[i] / iteration_time
'''

print(mean_acc)
print(mean_var)

# 推移をプロット
fig = plt.figure() #親グラフと子グラフを同時に定義

ax1 = fig.add_subplot(1, 1, 1)
ax1.set_xlabel('threshold')

# ax2 = ax1.twinx()
clrs = ['b', 'orange'] 
x = np.array(threshold)
acc_height = np.array(mean_acc)
var_height = np.array(mean_var)
# ax1.plot(x, var_height, color='blue')

ax1.plot(x, acc_height, color='red')
# ax1.plot(x, x, color='orange')

# plt.ylim(0.6, 0.75)
# ax.plot(left, var_height)
# plt.savefig("irt-all-e1.png")
plt.show()

