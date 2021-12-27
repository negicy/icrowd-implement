import pandas as pd
import numpy as np
from scipy import linalg


# jaccard係数で類似度計算
def jaccard_similarity_coeficient(list_a, list_b):
    # set a, bの積集合作成
    set_intersection = set.intersection(set(list_a), set(list_b))
    # set a, bの積集合の要素数取得
    num_intersection = len(set_intersection)

    # set a, bの和集合作成
    set_union = set.union(set(list_a), set(list_b))
    num_union = len(set_union)

    try:
        return float(num_intersection) / num_union
    except ZeroDivisionError:
        return 1.0

def split_text(text):
    # stop wordsを定義
    stop_words = set('for a of the and to in'.split())
    word_list = [word for word in text.lower().split() if word not in stop_words]

    return word_list

def generate_similarity_graph(label_df, task_list):
    # 類似度行列の作成
    size = len(task_list)
    similarity_matrix = np.zeros((size, size))
   
    threshold = 0.01

    for i_a in range(0, len(task_list)):
        jaccard_list = []
        task_id = task_list[i_a]
        text_a = label_df['title'][task_id]
        list_a = split_text(text_a)
        # list_a = []
        # est_label_a = label_df['estimate_label'][task_id]
        # list_a.append(est_label_a)
        
    
        for i_b in range(0, len(task_list)):
            task_id = task_list[i_b]
            text_b = label_df['title'][task_id]
            list_b = split_text(text_b)
            # list_b = []
            # est_label_b = label_df['estimate_label'][task_id]
            # list_b.append(est_label_b)

            jaccard = jaccard_similarity_coeficient(list_a, list_b)
            if jaccard >= threshold:
                similarity_matrix[(i_a, i_b)] = jaccard
            else:
                similarity_matrix[(i_a, i_b)] = 0

    return similarity_matrix
            

# icrowd (4)式
def page_rank(norm_sim_matrix, p, q, a):
    p = np.dot((1 / (1 + a)) * p, norm_sim_matrix) + (a / (1 + a)) * q
    return p

def normalize_sim(sim_matrix):

    d = np.diag(np.sum(sim_matrix, axis=0))
    a = np.sqrt(linalg.pinv(d))
    b = np.dot(sim_matrix, np.sqrt(linalg.pinv(d)))
    
    norm_sim_matrix = np.dot(a, b)

    return norm_sim_matrix



    

