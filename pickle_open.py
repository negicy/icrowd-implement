import pickle

d = {}
with open("main_objects.pickle", mode="rb") as f:
    d = pickle.load(f)

print(d)
