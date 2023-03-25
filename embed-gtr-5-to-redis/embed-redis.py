import os
import datetime
import json
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
from torch.hub import _get_torch_home
model = SentenceTransformer("sentence-transformers/gtr-t5-large")
print("model on fs is located here: " + _get_torch_home())

# what to embded
folder_to_scan = sys.argv[1] if len(sys.argv) > 1 else "."

batch_size = 32

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


#### create redis index, load all relevant settings, etc
# Constants
VECTOR_DIM = 768 # length of the vectors
VECTOR_NUMBER = 100                 # initial number of vectors
INDEX_NAME = "embeddings-index"                 # name of the search index
PREFIX = "doc"                                  # prefix for the document keys
DISTANCE_METRIC = "COSINE"                      # distance metric for the vectors (ex. COSINE, IP, L2)
# Define RediSearch fields for each of the columns in the dataset
source_file = TextField(name="source_file")
text = TextField(name="text")
text_embedding = VectorField("text_embedding",
    "FLAT", {
        "TYPE": "FLOAT32",
        "DIM": VECTOR_DIM,
        "DISTANCE_METRIC": DISTANCE_METRIC,
        "INITIAL_CAP": VECTOR_NUMBER,
    }
)
fields = [source_file, text, text_embedding]
# Check if index exists
try:
    redis_client.ft(INDEX_NAME).info()
    print("Index already exists")
except:
    # Create RediSearch Index
    redis_client.ft(INDEX_NAME).create_index(
        fields = fields,
        definition = IndexDefinition(prefix=[PREFIX], index_type=IndexType.HASH)
    )








def parse_file_content_to_chunks(file_content):
    content = file_content
    # split content by double newlines
    paragraphs = content.split("==")
    # paragraphs = [content]
    # trim and skip empty lines
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return paragraphs

def load_data(folder_to_scan):
    # recursively go over all md files in a folder and store their content in an array called texts
    file_count = 0
    for root, dirs, files in os.walk(folder_to_scan):
        for file in files:
            if file.endswith(".md") or file.endswith(".adoc") or file.endswith(".txt") or file.endswith(".org") or file.endswith(".tex"):
                file_count += 1
    texts = []
    source_files = []
    counter = 0
    batch_counter = 0
    for root, dirs, files in os.walk(folder_to_scan):
        for file in files:
            if file.endswith(".md") or file.endswith(".adoc") or file.endswith(".txt") or file.endswith(".org") or file.endswith(".tex"):
                counter += 1
                print("processing file " + str(counter) + " of " + str(file_count) + ": " + file)
                with open(os.path.join(root, file), "r") as f:
                    paragraphs = parse_file_content_to_chunks(f.read())
                    # append paragraphs array to texts array
                    texts.extend(paragraphs)
                    # append file multiple times to source_files array - so that a paragraph[i] corresponds to source_files[i]
                    source_files.extend(len(paragraphs)*[file])
                    batch_counter += len(paragraphs)
                    if (batch_counter > batch_size):
                        yield texts, source_files
                        texts = []
                        source_files = []
                        batch_counter = 0
    yield texts, source_files

def get_embeddings(texts):
    print("embedding starts, with no of texts: " + str(len(texts)))
    print(datetime.datetime.now().isoformat())
    embeddings = model.encode(texts)
    print(datetime.datetime.now().isoformat())
    print("embedding ends")
    print("embeddings.shape: " + str(embeddings.shape))
    print("embeddings[0].shape: " + str(embeddings[0].shape))
    print("embeddings[1].shape: " + str(embeddings[1].shape))
    return embeddings

def save_embeddings(texts, embeddings, source_files):
    # print("saving embeddings")
    # print(datetime.datetime.now().isoformat())
    # with open("embeddings.json", "a") as fp:
    #     json.dump(
    #         {
    #             "source_files": source_files,
    #             "embeddings": [list(map(float, e)) for e in embeddings],
    #             "texts": texts
    #         },
    #         fp,
    #     )
    # print(datetime.datetime.now().isoformat())
    # print("embeddings saved")
    for i in range(len(texts)):
        redis_client.hset(PREFIX + ':' + str(i), mapping =
            {
                "source_file": source_files[i],
                "text_embedding": np.array(embeddings[i], dtype=np.float32).tobytes(),
                "text": texts[i]
            }
        )

for texts, source_files in load_data(folder_to_scan):
    print("Performing chunked embedding on " + str(len(texts)) + " texts")
    embeddings = get_embeddings(texts)
    save_embeddings(texts, embeddings, source_files)
