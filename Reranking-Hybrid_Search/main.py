from fastembed import TextEmbedding, LateInteractionTextEmbedding, SparseTextEmbedding
from qdrant_client.models import Distance, VectorParams, models, PointStruct
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
print("Clients initialized")

dense_embedding_model = TextEmbedding(
    "sentence-transformers/all-MiniLM-L6-v2",
    cache_dir=r"C:\Users\Umberto\git\RAG-tutorials\Reranking-Hybrid_Search\.cache",
)
bm25_embedding_model = SparseTextEmbedding(
    "Qdrant/bm25",
    cache_dir=r"C:\Users\Umberto\git\RAG-tutorials\Reranking-Hybrid_Search\.cache",
)
late_interaction_embedding_model = LateInteractionTextEmbedding(
    "colbert-ir/colbertv2.0",
    cache_dir=r"C:\Users\Umberto\git\RAG-tutorials\Reranking-Hybrid_Search\.cache",
)


def load_data():
    documents = [
        "In machine learning, feature scaling is the process of normalizing the range of independent variables or features. The goal is to ensure that all features contribute equally to the model, especially in algorithms like SVM or k-nearest neighbors where distance calculations matter.",
        "Feature scaling is commonly used in data preprocessing to ensure that features are on the same scale. This is particularly important for gradient descent-based algorithms where features with larger scales could disproportionately impact the cost function.",
        "In data science, feature extraction is the process of transforming raw data into a set of engineered features that can be used in predictive models. Feature scaling is related but focuses on adjusting the values of these features.",
        "Unsupervised learning algorithms, such as clustering methods, may benefit from feature scaling as it ensures that features with larger numerical ranges don't dominate the learning process.",
        "One common data preprocessing technique in data science is feature selection. Unlike feature scaling, feature selection aims to reduce the number of input variables used in a model to avoid overfitting.",
        "Principal component analysis (PCA) is a dimensionality reduction technique used in data science to reduce the number of variables. PCA works best when data is scaled, as it relies on variance which can be skewed by features on different scales.",
        "Min-max scaling is a common feature scaling technique that usually transforms features to a fixed range [0, 1]. This method is useful when the distribution of data is not Gaussian.",
        "Standardization, or z-score normalization, is another technique that transforms features into a mean of 0 and a standard deviation of 1. This method is effective for data that follows a normal distribution.",
        "Feature scaling is critical when using algorithms that rely on distances, such as k-means clustering, as unscaled features can lead to misleading results.",
        "Scaling can improve the convergence speed of gradient descent algorithms by preventing issues with different feature scales affecting the cost function's landscape.",
        "In deep learning, feature scaling helps in stabilizing the learning process, allowing for better performance and faster convergence during training.",
        "Robust scaling is another method that uses the median and the interquartile range to scale features, making it less sensitive to outliers.",
        "When working with time series data, feature scaling can help in standardizing the input data, improving model performance across different periods.",
        "Normalization is often used in image processing to scale pixel values to a range that enhances model performance in computer vision tasks.",
        "Feature scaling is significant when features have different units of measurement, such as height in centimeters and weight in kilograms.",
        "In recommendation systems, scaling features such as user ratings can improve the model's ability to find similar users or items.",
        "Dimensionality reduction techniques, like t-SNE and UMAP, often require feature scaling to visualize high-dimensional data in lower dimensions effectively.",
        "Outlier detection techniques can also benefit from feature scaling, as they can be influenced by unscaled features that have extreme values.",
        "Data preprocessing steps, including feature scaling, can significantly impact the performance of machine learning models, making it a crucial part of the modeling pipeline.",
        "In ensemble methods, like random forests, feature scaling is not strictly necessary, but it can still enhance interpretability and comparison of feature importance.",
        "Feature scaling should be applied consistently across training and test datasets to avoid data leakage and ensure reliable model evaluation.",
        "In natural language processing (NLP), scaling can be useful when working with numerical features derived from text data, such as word counts or term frequencies.",
        "Log transformation is a technique that can be applied to skewed data to stabilize variance and make the data more suitable for scaling.",
        "Data augmentation techniques in machine learning may also include scaling to ensure consistency across training datasets, especially in computer vision tasks.",
    ]

    dense_embeddings = list(dense_embedding_model.embed(doc for doc in documents))
    bm25_embeddings = list(bm25_embedding_model.embed(doc for doc in documents))
    late_interaction_embeddings = list(
        late_interaction_embedding_model.embed(doc for doc in documents)
    )

    # Create Collection
    client.create_collection(
        "hybrid-search",
        vectors_config={
            "all-MiniLM-L6-v2": models.VectorParams(
                size=len(dense_embeddings[0]),
                distance=models.Distance.COSINE,
            ),
            "colbertv2.0": models.VectorParams(
                size=len(late_interaction_embeddings[0][0]),
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                ),
            ),
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )

    # Upsert Data
    points = []
    for idx, (
        dense_embedding,
        bm25_embedding,
        late_interaction_embedding,
        doc,
    ) in enumerate(
        zip(dense_embeddings, bm25_embeddings, late_interaction_embeddings, documents)
    ):

        point = PointStruct(
            id=idx,
            vector={
                "all-MiniLM-L6-v2": dense_embedding,
                "bm25": bm25_embedding.as_object(),
                "colbertv2.0": late_interaction_embedding,
            },
            payload={"document": doc},
        )
        points.append(point)

    operation_info = client.upsert(collection_name="hybrid-search", points=points)


def retrieval():

    query = "What is the purpose of feature scaling in machine learning?"

    dense_vectors = next(dense_embedding_model.query_embed(query))
    sparse_vectors = next(bm25_embedding_model.query_embed(query))
    late_vectors = next(late_interaction_embedding_model.query_embed(query))

    prefetch = [
        models.Prefetch(
            query=dense_vectors,
            using="all-MiniLM-L6-v2",
            limit=20,
        ),
        models.Prefetch(
            query=models.SparseVector(**sparse_vectors.as_object()),
            using="bm25",
            limit=20,
        ),
    ]

    results = client.query_points(
        "hybrid-search",
        prefetch=prefetch,
        query=late_vectors,
        using="colbertv2.0",
        with_payload=True,
        limit=10,
    )

    print(results)


if __name__ == "__main__":

    # load_data()
    retrieval()
