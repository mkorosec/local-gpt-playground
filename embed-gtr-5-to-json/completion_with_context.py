import os
import datetime
import json
import sys
import subprocess

# input string from first argument, exit if not provided
if len(sys.argv) < 2:
    print("Usage: " + sys.argv[0] + " <query>")
    sys.exit(1)

query = sys.argv[1]

# encode the query into an embedding vector
from sentence_transformers import SentenceTransformer
from torch.hub import _get_torch_home
model = SentenceTransformer("sentence-transformers/gtr-t5-large")
query_embedding = model.encode(query)

# load the embeddings
import faiss
import numpy as np
data = json.load(open("embeddings.json"))
index = faiss.IndexFlatL2(len(data["embeddings"][0]))
index.add(np.array(data["embeddings"]))

# perform the search
D, I = index.search(np.array([query_embedding]), 5)

# print the results
context = []
for i in range(len(I[0])):
    ix = I[0][i]
    result = data["source_files"][ix]+": "+data["texts"][ix]+"\n"
    context.append(result)

# completion prompt: convert the array to string by concatenating each element with a newline
completion_prompt_context = "You will receive a query to be answered in a specific context. First, the context will be given, with reference files and relevant excerpts. Then the query will follow. Answer the query by taking the context into account. Context:\n"
completion_prompt_query = "\nQuery: " + query
completion_prompt = completion_prompt_context + '\n\n'.join(context) + completion_prompt_query

# replace all newlines with spaces
completion_prompt = completion_prompt.replace('\n', ' ')
print(completion_prompt)
