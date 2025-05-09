Create the environment **Reranking-Hybrid_Search**:

```shell
# Create and activate the env with conda
conda create --name=Reranking-Hybrid_Search python=3.10 -y && conda activate Reranking-Hybrid_Search && pip install uv

# Install production libs
uv pip install fastembed==0.6.1 qdrant-client==1.14.2
```

Remove the **Reranking-Hybrid_Search** environment:
```shell
conda deactivate && conda remove --name=Reranking-Hybrid_Search --all -y
```

LINK TUTORIAL -> https://qdrant.tech/documentation/search-precision/reranking-hybrid-search/



```shell
# QDRANT
docker run -p 6333:6333 -p 6334:6334 -v "qdrant_storage:/qdrant/storage:z" qdrant/qdrant
```