import os
import datetime
import json
import sys
from sentence_transformers import SentenceTransformer
from torch.hub import _get_torch_home
model = SentenceTransformer("sentence-transformers/gtr-t5-large")
print("model on fs is located here: " + _get_torch_home())

# what to embded
folder_to_scan = sys.argv[1] if len(sys.argv) > 1 else "."

def parse_file_content_to_chunks(file_content):
    content = file_content
    # split content by double newlines
    paragraphs = content.split("\n\n")
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
    return texts, source_files

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

def save_embeddings(embeddings, source_files):
    print("saving embeddings")
    print(datetime.datetime.now().isoformat())
    with open("embeddings.json", "w") as fp:
        json.dump(
            {
                "source_files": source_files,
                "embeddings": [list(map(float, e)) for e in embeddings],
                "texts": texts
            },
            fp,
        )
    print(datetime.datetime.now().isoformat())
    print("embeddings saved")

texts, source_files = load_data(folder_to_scan)
embeddings = get_embeddings(texts)
save_embeddings(embeddings, source_files)
