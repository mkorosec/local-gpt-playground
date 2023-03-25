# local-gpt-playground

## setup

```
python3 -m venv env
source env/bin/activate 
pip install sentence-transformers 
pip install httpx faiss-cpu
```

## build index

```
python embed-gtr-5-to-redis/embded-redis.py <folder>
```

* will recursively scan folder for txt/md/org/tex files
* split the text by \n\n
* calculate embeddings for each paragraph
* store an entry to redis with a reference to source_file, paragraph (full text), embedding vector

## query embeddings

```
python embed-gtr-5-to-redis/completion_with_context-redis.py '<query>'
```

* search redis with knn
* calculate embedding vector for query
* find 5 nearest neighbours in redis from the embedded vectors
* print them to stdout

# TODO

* query alpaca with context gained from embeddings + query
+ introduce a vector db instead of a json file

* UI module that can search the redis db for similar chunks -> display source_file + relevant part of text


# Related resources

https://simonwillison.net/2023/Jan/13/semantic-search-answers/
https://til.simonwillison.net/python/gtr-t5-large
https://huggingface.co/sentence-transformers/gtr-t5-large
https://github.com/antimatter15/alpaca.cpp
https://github.com/openai/openai-cookbook/tree/main/examples/vector_databases
