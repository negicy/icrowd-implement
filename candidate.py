import math
import girth
import pandas as pd
import numpy as np
import random


# qualification task, test taskに分けてシミュレーション
# IRT: girthを使用
def make_candidate(threshold, input_df, p_vector, worker_list, task_list):
    worker_c_th = {}
    task_id_dic = {}
    for i in range(0, len(task_list)):
        task_id_dic[task_list[i]] = i
    # 承認タスクとテストタスクを分離
    random.shuffle(task_list)
    qualify_task = task_list[:50]
    test_task = task_list[50:]
    test_worker = random.sample(worker_list, 20)

    # for test in test_

    # t_worker = worker_list
    p_w_dic = {}
    q_w_list = []

    for worker in test_worker:
        q_w = np.zeros(len(task_list))
        for i in range(0, len(task_list)):
            task_id = task_list[i]
            if task_id in qualify_task:
                q_w[i] = input_df[worker][task_id]
        # q_w_list.append(q_w)
        # for worker in test_worker:
        p_w = 0
        # p_t: vector, q: schalar
        for i in range(0, len(p_vector)):
            p_w += q_w[i]*p_vector[i]
        p_w_dic[worker] = p_w
                
    
    for th in threshold:   
        #p_w.append(p_t)
        # print(len(p_t))
        # p_w_list.append(p_w)

        # output: worker_c = {task: []}
        # すべてのスレッショルドについてワーカー候補作成
        
        candidate_count = 0
        worker_c = {}
        for task in test_task:
            worker_c[task] = []
            # workerの正答確率prob
            # worker_c
            index = task_id_dic[task]
            for worker in test_worker:
                
                prob = p_w_dic[worker][index]           
                # workerの正解率がthresholdより大きければ
                if prob >= th:
                    # ワーカーを候補リストに代入
                    worker_c[task].append(worker)
    
        worker_c_th[th] = worker_c

    return worker_c_th, test_worker, test_task


def Frequency_Distribution(data, lim, class_width=None):
    data = np.asarray(data)
    if class_width is None:
        class_size = int(np.log2(data.size).round()) + 1
        class_width = round((data.max() - data.min()) / class_size)

    bins = np.arange(lim[0], lim[1], class_width)
    print(bins)
    hist = np.histogram(data, bins)[0]
    cumsum = hist.cumsum()

    return pd.DataFrame({'階級値': (bins[1:] + bins[:-1]) / 2,
                         '度数': hist,
                         '累積度数': cumsum,
                         '相対度数': hist / cumsum[-1],
                         '累積相対度数': cumsum / cumsum[-1]},
                        index=pd.Index([f'{bins[i]}以上{bins[i+1]}未満'
                                        for i in range(hist.size)],
                                       name='階級'))

