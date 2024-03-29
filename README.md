# About

This repository contains code for nlmatics code for indexing, retrieval, extraction and generation.

## Installation instructions

### Run the docker file
A docker image is available via public github container registry. 

Pull the docker image
```
docker pull ghcr.io/nlmatics/nlm-discovery-engine:latest
```
Run the docker image mapping the port 5001 to port of your choice. 
```
docker run -p 5011:5001 ghcr.io/nlmatics/nlm-discovery-engine:latest
```

## Indexing Algorithm

- Use nlm-ingestor to break down a document with layout information
- Index each line of the the document and add the following information to the index:
    1. The keywords in the sentence
    2. The entity types in the sentence e.g. NUM: percentage, ENT: Org etc.
    3. The header chain of the sentence
    4. The keywords in the paragraph of the sentence

- Indexing is done with elasticsearch
- A embedding index is also added to the elastic search with SIF as the default embeddingn and DPR as an option

## Retreival Algorithm

- For keyword retrievals use:
    1. prefer tri-grams > bi-gram > presence of one or more matching words > sif
- For question retrievals use:
    1. use return type > prefer tri-grams > bi-gram > presence of one or more matching words > sif

- Based on the task, use one of the models in nlm-model-service to evaluate the retrieved passages.

## Credits

This code was developed at Nlmatics Corp. from 2020-2023.

Yi Zhang wrote the majority of the pipeline. Reshav Abraham added code for answer type prioritization. Ambika Sukla added code for efficient batching of embeddings, topic processor, relation extraction processor and dynamic value processor. Kiran Panicker optimized the pipeline, added workspace ranking logic and abstractive summary processor.