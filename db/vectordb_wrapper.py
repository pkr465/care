# vectordb_wrapper.py

import logging
from db.postgres_api import PostgresVectorStore

class VectorDB:
    """
    High-level interface for vector store operations, supporting Postgres backends.
    """
    def __init__(self, env_config):
        """
        Initialize based on env_config and instantiate the underlying store.
        Args:
            env_config (dict): Environment and connection parameters
        """
        self.env_config = env_config
        self.logger = logging.getLogger(__name__)
        self.vector_db = env_config.get("VECTOR_DATABASE", "postgres").lower()
        self.vector_store = self._initialize_store()

    def _initialize_store(self):
        """
        Create and return an instance of HermesVectorStore or PostgresVectorStore.
        """
        db = self.vector_db
        if db == "postgres":
            pg_vector_store = PostgresVectorStore(
                self.env_config.get("POSTGRES_CONNECTION"),
                self.env_config.get("POSTGRES_COLLECTION"),
                self.env_config.get("POSTGRES_EMBEDDING_TABLENAME"),
                self.env_config.get("POSTGRES_COLLECTION_EMBEDDING_TABLENAME"),
                self.env_config.get("QGENIE_API_KEY")
            )
            return pg_vector_store
        else:
            raise ValueError(f"Unknown vector backend: {db}")

    def store_embeddings(self, docs):
        """
        Store embeddings for a list of documents.
        Args:
            docs (list[Document]): Documents to embed and store.
        Returns:
            vector store instance or None
        """
        if self.vector_store is None:
            raise RuntimeError("Vector store not initialized")
        return self.vector_store.store_embeddings(docs)

    def retrieve(self, query, k=2, threshold=0.0, similarity_threshold=False):
        """
        Retrieve most similar documents to the given query.
        Args:
            query (str): Text query.
            k (int): Number of results.
            threshold (float): Optional similarity threshold.
            similarity_threshold (bool): Use threshold or not.
        Returns:
            list[Document]
        """
        if self.vector_store is None:
            raise RuntimeError("Vector store not initialized")
        return self.vector_store.retrieve(query, k, threshold, similarity_threshold)
    
    def run_custom_query(self, sql):
        """
        Execute a custom SQL query, if supported by the backend.
        Args:
            sql (str): SQL statement to execute (typically for Postgres).
        Returns:
            list[dict]: Query results.
        """
        if self.vector_store is None:
            raise RuntimeError("Vector store not initialized")
        if self.vector_db == "postgres":
            return self.vector_store.run_custom_query(sql)
        else:
            raise NotImplementedError("Custom SQL queries are only supported for the Postgres backend.")
      