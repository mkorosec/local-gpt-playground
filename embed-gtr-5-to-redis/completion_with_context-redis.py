import os
import datetime
import json
import sys
import subprocess
import numpy as np

# input string from first argument, exit if not provided
if len(sys.argv) < 2:
    print("Usage: " + sys.argv[0] + " <query>")
    sys.exit(1)

query = sys.argv[1]




# init redis client
import redis
from redis.commands.search.indexDefinition import (
    IndexDefinition,
    IndexType
)
from redis.commands.search.query import Query
from redis.commands.search.field import (
    TextField,
    VectorField
)
REDIS_HOST =  "localhost"
REDIS_PORT = 6379
REDIS_PASSWORD = "" # default for passwordless Redis
# Connect to Redis
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD
)
redis_client.ping()





# init gtr-t5-large model - for query embedding
from sentence_transformers import SentenceTransformer
from torch.hub import _get_torch_home
model = SentenceTransformer("sentence-transformers/gtr-t5-large")






# init - load up embeddings db
def search_redis(
    redis_client: redis.Redis,
    user_query: str,
    index_name: str = "embeddings-index",
    vector_field: str = "text_embedding",
    return_fields: list = ["source_file", "text", "vector_score"],
    hybrid_fields = "*",
    k: int = 20,
):
    # Creates embedding vector from user query
    # encode the query into an embedding vector
    query_embedding = model.encode(user_query)
    # Prepare the Query
    base_query = f'{hybrid_fields}=>[KNN {k} @{vector_field} $vector AS vector_score]'
    redis_query = (
        Query(base_query)
         .return_fields(*return_fields)
         .sort_by("vector_score")
         .paging(0, k)
         .dialect(2)
    )
    params_dict = {"vector": np.array(query_embedding).astype(dtype=np.float32).tobytes()}
    # perform vector search
    results = redis_client.ft(index_name).search(redis_query, params_dict)
    for i, article in enumerate(results.docs):
        score = 1 - float(article.vector_score)
        print(f"{i}. {article.text}... (Score: {round(score ,3) }, Source: {article.source_file})")
    return results.docs



results = search_redis(redis_client, query, k=10)

# # completion prompt: convert the array to string by concatenating each element with a newline
# completion_prompt_context = "You will receive a query to be answered in a specific context. First, the context will be given, with reference files and relevant excerpts. Then the query will follow. Answer the query by taking the context into account. Context:\n"
# completion_prompt_query = "\nQuery: " + query
# completion_prompt = completion_prompt_context + '\n\n'.join(context) + completion_prompt_query
# 
# # replace all newlines with spaces
# completion_prompt = completion_prompt.replace('\n', ' ')
# print(completion_prompt)
