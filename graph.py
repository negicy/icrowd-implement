import pandas as pd
import numpy as np
import pickle

label_df = pd.read_csv("label_df.csv", sep = ",")
batch_df = pd.read_csv("batch_100.csv", sep = ",")
label_df = label_df.set_index('id')


d = {}
with open("main_objects.pickle", mode="rb") as f:
    d = pickle.load(f)

print(d)
worker_list = d['worker_list']
task_list = d['task_list']
input_df = d['input_df']

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
   
    threshold = 0.1

    for i_a in range(0, len(task_list)):
        jaccard_list = []
        task_id = task_list[i_a]
        text_a = label_df['title'][task_id]
        list_a = split_text(text_a)
        # est_label_a = label_df['estimate_label'][id_a]
        # list_a.append(est_label_a)
        
        print(list_a)
        for i_b in range(0, len(task_list)):
            task_id = task_list[i_b]
            text_b = label_df['title'][task_id]
            list_b = split_text(text_b)
            # est_label_b = label_df['estimate_label'][id_b]
            # list_b.append(est_label_b)

            jaccard = 0
            if jaccard >= threshold:
                jaccard = jaccard_similarity_coeficient(list_a, list_b)
        similarity_matrix[(i_a, i_b)] = jaccard
    return similarity_matrix
            
s = generate_similarity_graph(label_df, task_list)
print(s)



#jaccard = jaccard_similarity_coeficient(list_a, list_b)
#print(jaccard)
