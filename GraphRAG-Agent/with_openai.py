#######################################
#######################################
# First, the user query is defined ("How is Bob connected to New York?").
# - The QdrantNeo4jRetriever searches for related entities in the
#   Qdrant vector database based on the user query's embedding.
#   It retrieves the top 5 results (top_k=5).
#
# - The entity_ids are extracted from the retriever result.
#
# - The fetch_related_graph function retrieves related
#   entities and their relationships from the Neo4j database.
#
# - The format_graph_context function prepares the graph
#   data in a format the LLM can understand.
#
# - Finally, the graphRAG_run function is called to generate
#   and query the language model, producing an answer
#   based on the retrieved graph context.
#
# With this, we have successfully created GraphRAG, a system capable
# of capturing complex relationships and delivering improved
# performance compared to the baseline RAG approach.
#######################################
#######################################

from neo4j import GraphDatabase
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI
from neo4j_graphrag.retrievers import QdrantNeo4jRetriever
import uuid
import os

# Load environment variables
load_dotenv()

# Get credentials from environment variables
# qdrant_key = os.getenv("QDRANT_KEY")
qdrant_url = os.getenv("QDRANT_URL")
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")
openai_organization = os.getenv("OPENAI_ORGANIZATION")
openai_key = os.getenv("OPENAI_API_KEY")


# Defining Output Parser
# These classes help ensure that data from the OpenAI
# LLM is parsed correctly into the graph components (nodes and relationships).
class single(BaseModel):
    node: str
    target_node: str
    relationship: str


class GraphComponents(BaseModel):
    graph: list[single]


# We now initialize the OpenAI client and define
# a function to send prompts to the LLM and parse its responses.
client = OpenAI(organization=openai_organization)


def openai_llm_parser(prompt):
    """
    This function sends a prompt to the LLM, asking it
    to extract graph components (nodes and relationships)
    from the provided text. The response is parsed
    into structured graph data.
    """

    completion = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": """ You are a precise graph relationship extractor. Extract all 
                    relationships from the text and format them as a JSON object 
                    with this exact structure:
                    {
                        "graph": [
                            {"node": "Person/Entity", 
                             "target_node": "Related Entity", 
                             "relationship": "Type of Relationship"},
                            ...more relationships...
                        ]
                    }
                    Include ALL relationships mentioned in the text, including 
                    implicit ones. Be thorough and precise. """,
            },
            {"role": "user", "content": prompt},
        ],
    )

    return GraphComponents.model_validate_json(completion.choices[0].message.content)


def extract_graph_components(raw_data):
    """
    This function takes raw data, uses the LLM to parse it into
    graph components, and then assigns unique IDs to nodes and relationships.
    """

    prompt = f"Extract nodes and relationships from the following text:\n{raw_data}"

    parsed_response = openai_llm_parser(
        prompt
    )  # Assuming this returns a list of dictionaries
    parsed_response = (
        parsed_response.graph
    )  # Assuming the 'graph' structure is a key in the parsed response

    nodes = {}
    relationships = []

    for entry in parsed_response:
        node = entry.node
        target_node = entry.target_node  # Get target node if available
        relationship = entry.relationship  # Get relationship if available

        # Add nodes to the dictionary with a unique ID
        if node not in nodes:
            nodes[node] = str(uuid.uuid4())

        if target_node and target_node not in nodes:
            nodes[target_node] = str(uuid.uuid4())

        # Add relationship to the relationships list with node IDs
        if target_node and relationship:
            relationships.append(
                {
                    "source": nodes[node],
                    "target": nodes[target_node],
                    "type": relationship,
                }
            )

    return nodes, relationships


def ingest_to_neo4j(nodes, relationships):
    """
    Ingest nodes and relationships into Neo4j.
    """

    with neo4j_driver.session() as session:
        # Create nodes in Neo4j
        for name, node_id in nodes.items():
            session.run(
                "CREATE (n:Entity {id: $id, name: $name})", id=node_id, name=name
            )

        # Create relationships in Neo4j
        for relationship in relationships:
            session.run(
                "MATCH (a:Entity {id: $source_id}), (b:Entity {id: $target_id}) "
                "CREATE (a)-[:RELATIONSHIP {type: $type}]->(b)",
                source_id=relationship["source"],
                target_id=relationship["target"],
                type=relationship["type"],
            )

    return nodes


def create_collection(client, collection_name, vector_dimension):
    """
    - Qdrant Client: The QdrantClient is used to connect to the Qdrant instance.
    - Creating Collection: The create_collection function checks
        if a collection exists. If not, it creates one with a
        specified vector dimension and distance metric
        (cosine similarity in this case).
    """

    # Try to fetch the collection status
    try:
        collection_info = client.get_collection(collection_name)
        print(f"Skipping creating collection; '{collection_name}' already exists.")

    except Exception as e:
        # If collection does not exist, an error will be thrown, so we create the collection
        if "Not found: Collection" in str(e):
            print(f"Collection '{collection_name}' not found. Creating it now...")

            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_dimension, distance=models.Distance.COSINE
                ),
            )

            print(f"Collection '{collection_name}' created successfully.")
        else:
            print(f"Error while checking collection: {e}")


def openai_embeddings(text):
    """
    This function uses OpenAI's embedding model to
    transform input text into vector representations.
    """

    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small",
    )

    return response.data[0].embedding


def ingest_to_qdrant(collection_name, raw_data, node_id_mapping):
    """
    The ingest_to_qdrant function generates embeddings for each
    paragraph in the raw data and stores them in a Qdrant
    collection. It associates each embedding with a unique ID
    and its corresponding node ID from the node_id_mapping
    dictionary, ensuring proper linkage for later retrieval.
    """
    embeddings = [openai_embeddings(paragraph) for paragraph in raw_data.split("\n")]

    qdrant_client.upsert(
        collection_name=collection_name,
        points=[
            {"id": str(uuid.uuid4()), "vector": embedding, "payload": {"id": node_id}}
            for node_id, embedding in zip(node_id_mapping.values(), embeddings)
        ],
    )


def retriever_search(neo4j_driver, qdrant_client, collection_name, query):
    """
    The QdrantNeo4jRetriever handles both vector search
    and graph data fetching, combining Qdrant for vector-based
    retrieval and Neo4j for graph-based queries.

    Vector Search:
        - qdrant_client connects to Qdrant for efficient vector similarity search.
        - collection_name specifies where vectors are stored.
        - id_property_external="id" maps the external entity's ID for retrieval.

    Graph Fetching:
        - neo4j_driver connects to Neo4j for querying graph data.
        - id_property_neo4j="id" ensures the entity IDs from
            Qdrant match the graph nodes in Neo4j.
    """
    retriever = QdrantNeo4jRetriever(
        driver=neo4j_driver,
        client=qdrant_client,
        collection_name=collection_name,
        id_property_external="id",
        id_property_neo4j="id",
    )

    results = retriever.search(query_vector=openai_embeddings(query), top_k=5)

    return results


def fetch_related_graph(neo4j_client, entity_ids):
    """
    We need to fetch subgraph data from a Neo4j database
    based on specific entity IDs after the retriever
    has provided the relevant IDs.

    The function fetch_related_graph takes in a Neo4j client
    and a list of entity_ids. It runs a Cypher query to find
    related nodes (entities) and their relationships based
    on the given entity IDs. The query matches
    entities (e:Entity) and finds related nodes through any
    relationship [r]. The function returns a list of
    subgraph data, where each record contains the
    entity, relationship, and related_node.
    """

    query = """
    MATCH (e:Entity)-[r1]-(n1)-[r2]-(n2)
    WHERE e.id IN $entity_ids
    RETURN e, r1 as r, n1 as related, r2, n2
    UNION
    MATCH (e:Entity)-[r]-(related)
    WHERE e.id IN $entity_ids
    RETURN e, r, related, null as r2, null as n2
    """
    with neo4j_client.session() as session:
        result = session.run(query, entity_ids=entity_ids)
        subgraph = []
        for record in result:
            subgraph.append(
                {
                    "entity": record["e"],
                    "relationship": record["r"],
                    "related_node": record["related"],
                }
            )
            if record["r2"] and record["n2"]:
                subgraph.append(
                    {
                        "entity": record["related"],
                        "relationship": record["r2"],
                        "related_node": record["n2"],
                    }
                )
    return subgraph


def format_graph_context(subgraph):
    """
    The function format_graph_context processes a subgraph
    returned by a Neo4j query. It extracts the graph's
    entities (nodes) and relationships (edges). The
    nodes set ensures each entity is added only once.
    The edges list captures the relationships in
    a readable format: Entity1 relationship Entity2.
    """

    nodes = set()
    edges = []

    for entry in subgraph:
        entity = entry["entity"]
        related = entry["related_node"]
        relationship = entry["relationship"]

        nodes.add(entity["name"])
        nodes.add(related["name"])

        edges.append(f"{entity['name']} {relationship['type']} {related['name']}")

    return {"nodes": list(nodes), "edges": edges}


def graphRAG_run(graph_context, user_query):
    """
    The function graphRAG_run takes the graph context (nodes and edges)
    and the user query, combining them into a structured prompt
    for the LLM. The nodes and edges are formatted as readable
    strings to form part of the LLM input.
    The LLM is then queried with the generated prompt, asking
    it to refine the user query using the graph context
    and provide an answer. If the model successfully generates a
    response, it returns the answer.
    """

    nodes_str = ", ".join(graph_context["nodes"])
    edges_str = "; ".join(graph_context["edges"])
    prompt = f"""
    You are an intelligent assistant with access to the following knowledge graph:

    Nodes: {nodes_str}

    Edges: {edges_str}

    Using this graph, Answer the following question:

    User Query: "{user_query}"
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {
                    "role": "system",
                    "content": "Provide the answer for the following question:",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message

    except Exception as e:
        return f"Error querying LLM: {str(e)}"


def extract_structured_data(qdrant_client, collection_name):
    print("Creating collection...")
    collection_name = "graphRAGstoreds"
    vector_dimension = 1536
    create_collection(qdrant_client, collection_name, vector_dimension)
    print("Collection created/verified")

    print("Extracting graph components...")

    raw_data = """Alice is a data scientist at TechCorp's Seattle office.
    Bob and Carol collaborate on the Alpha project.
    Carol transferred to the New York office last year.
    Dave mentors both Alice and Bob.
    TechCorp's headquarters is in Seattle.
    Carol leads the East Coast team.
    Dave started his career in Seattle.
    The Alpha project is managed from New York.
    Alice previously worked with Carol at DataCo.
    Bob joined the team after Dave's recommendation.
    Eve runs the West Coast operations from Seattle.
    Frank works with Carol on client relations.
    The New York office expanded under Carol's leadership.
    Dave's team spans multiple locations.
    Alice visits Seattle monthly for team meetings.
    Bob's expertise is crucial for the Alpha project.
    Carol implemented new processes in New York.
    Eve and Dave collaborated on previous projects.
    Frank reports to the New York office.
    TechCorp's main AI research is in Seattle.
    The Alpha project revolutionized East Coast operations.
    Dave oversees projects in both offices.
    Bob's contributions are mainly remote.
    Carol's team grew significantly after moving to New York.
    Seattle remains the technology hub for TechCorp."""

    nodes, relationships = extract_graph_components(raw_data)
    print("Nodes:", nodes)
    print("Relationships:", relationships)

    print("Ingesting to Neo4j...")
    node_id_mapping = ingest_to_neo4j(nodes, relationships)
    print("Neo4j ingestion complete")

    print("Ingesting to Qdrant...")
    ingest_to_qdrant(collection_name, raw_data, node_id_mapping)
    print("Qdrant ingestion complete")


def get_answer(query, neo4j_driver, qdrant_client, collection_name):
    print("Starting retriever search...")
    retriever_result = retriever_search(
        neo4j_driver, qdrant_client, collection_name, query
    )
    print("Retriever results:", retriever_result)

    print("Extracting entity IDs...")
    entity_ids = [
        item.content.split("'id': '")[1].split("'")[0]
        for item in retriever_result.items
    ]
    print("Entity IDs:", entity_ids)

    print("Fetching related graph...")
    subgraph = fetch_related_graph(neo4j_driver, entity_ids)
    print("Subgraph:", subgraph)

    print("Formatting graph context...")
    graph_context = format_graph_context(subgraph)
    print("Graph context:", graph_context)

    print("Running GraphRAG...")
    answer = graphRAG_run(graph_context, query)
    print("Final Answer:", answer)


if __name__ == "__main__":
    print("Script started")
    print("Loading environment variables...")
    load_dotenv(".env.local")
    print("Environment variables loaded")

    print("Initializing clients...")
    neo4j_driver = GraphDatabase.driver(
        neo4j_uri, auth=(neo4j_username, neo4j_password)
    )
    qdrant_client = QdrantClient(url=qdrant_url)
    print("Clients initialized")

    collection_name = "graphRAGstoreds"
    #extract_structured_data(qdrant_client, collection_name)

    query = "How is Bob connected to New York?"
    get_answer(query, neo4j_driver, qdrant_client, collection_name)
