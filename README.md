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

## query embeddings

```
python find.py '<query>'
```

# TODO

* setup alpaca.cpp
* build a "context" query from embeddings
* query alpaca with context gained from embeddings + query