import pickle
with open("bm25_index.pkl","rb") as f:
    data=pickle.load(f)
print(type(data))
print(data)    