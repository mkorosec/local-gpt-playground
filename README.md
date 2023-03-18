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
python embded.py <folder>
```

* will recursively scan folder for txt/md/org/tex files
* split the text by \n\n
* calculate embeddings for each paragraph
* store a json file with a reference to source_file, paragraph (full text), embedding vector

## query embeddings

```
python find.py '<query>'
```

* load the json file dumped from the previous step
* calculate embedding vector for <query>
* find 5 nearest neighbours
* print them to stdout

# TODO

* setup alpaca.cpp
* build a "context" query from embeddings
* query alpaca with context gained from embeddings + query
* introduce a vector db instead of a json file

# Related resources

https://simonwillison.net/2023/Jan/13/semantic-search-answers/
https://til.simonwillison.net/python/gtr-t5-large
https://huggingface.co/sentence-transformers/gtr-t5-large
https://github.com/antimatter15/alpaca.cpp
https://github.com/openai/openai-cookbook/tree/main/examples/vector_databases
