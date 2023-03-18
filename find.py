import os
import datetime
import json
import sys

# input string from first argument, exit if not provided
if len(sys.argv) < 2:
    print("Usage: " + sys.argv[0] + " <query>")
    sys.exit(1)

query = sys.argv[1]

# encode the query into an embedding vector
from sentence_transformers import SentenceTransformer
from torch.hub import _get_torch_home
model = SentenceTransformer("sentence-transformers/gtr-t5-large")
print("model on fs is located here: " + _get_torch_home())

query_embedding = model.encode(query)

print("query: " + query)
print("query_embedding.shape: " + str(query_embedding.shape))


# load the embeddings
import faiss
import numpy as np
data = json.load(open("embeddings.json"))
index = faiss.IndexFlatL2(len(data["embeddings"][0]))
index.add(np.array(data["embeddings"]))

# perform the search
D, I = index.search(np.array([query_embedding]), 5)
print("D.shape: " + str(D.shape))
print("I.shape: " + str(I.shape))
print(D)

# iterate over array and get index and value
    

# print the results
for i in range(len(I[0])):
    ix = I[0][i]
    print("=== Match " + str(i) + " (distance: "+ str(D[0][i]) +") ===\n" + data["source_files"][ix]+": '"+data["texts"][ix]+"'\n=====================================================\n")
