import json
import uuid
import hashlib
from typing import List
import pandas as pd
from vanna import VannaBase
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    ComplexField,
    CorsOptions,
    SearchIndex,
    ScoringProfile,
    SearchFieldDataType,
    SimpleField,
    SearchableField
)


class AzureAISearch(VannaBase):
    
    def __init__(self, config=None):
        VannaBase.__init__(self, config=config)
        if config is None:
            config = {}

        key = config.get("key")
        endpoint = config.get("endpoint")
        prefix = config.get("prefix")
        index_sql = f"{prefix}_1"
        index_doc = f"{prefix}_2"
        index_ddl = f"{prefix}_3"
        credential = AzureKeyCredential(key)
        search_index_client = SearchIndexClient(endpoint, credential)
        scoring_profiles: List[ScoringProfile] = []
        ################################################################
        self.search_client_sql = search_index_client.get_index(index_sql)
        if self.search_client_sql is None:
          fields = [
              SearchableField(name="question", type=SearchFieldDataType.String, key=True),
              SearchableField(name="sql", type=SearchFieldDataType.String),
          ]
          self.search_client_sql = search_index_client.create_index(SearchIndex(name=index_sql, fields=fields, scoring_profiles=scoring_profiles))
        ################################################################
        self.search_client_doc = search_index_client.get_index(index_doc)
        if self.search_client_doc is None:
          fields = [
              SearchableField(name="documentation", type=SearchFieldDataType.String, key=True)
          ]
          self.search_client_sql = search_index_client.create_index(SearchIndex(name=index_doc, fields=fields, scoring_profiles=scoring_profiles))
        ################################################################
        self.search_client_sql = search_index_client.get_index(index_ddl)
        if self.search_client_ddl is None:
          fields = [
              SearchableField(name="ddl", type=SearchFieldDataType.String, key=True)
          ]
          self.search_client_sql = search_index_client.create_index(SearchIndex(name=index_ddl, fields=fields, scoring_profiles=scoring_profiles))

    def create_uuid_from_string(val: str):
      hex_string = hashlib.md5(val.encode("UTF-8")).hexdigest()
      return uuid.UUID(hex=hex_string)

    def generate_embedding(self, data: str, **kwargs) -> List[float]:
        embedding = self.embedding_function([data])
        if len(embedding) == 1:
            return embedding[0]
        return embedding

    def add_question_sql(self, question: str, sql: str, **kwargs) -> str:
        question_sql_json = json.dumps(
            {
                "question": question,
                "sql": sql,
            },
            ensure_ascii=False,
        )
        id = self.create_uuid_from_string(question_sql_json) + "-sql"
        self.sql_collection.add(
            documents=question_sql_json,
            embeddings=self.generate_embedding(question_sql_json),
            ids=id,
        )

        return id

    def add_ddl(self, ddl: str, **kwargs) -> str:
        id = self.create_uuid_from_string(ddl) + "-ddl"
        self.ddl_collection.add(
            documents=ddl,
            embeddings=self.generate_embedding(ddl),
            ids=id,
        )
        return id

    def add_documentation(self, documentation: str, **kwargs) -> str:
        id = self.create_uuid_from_string(documentation) + "-doc"
        self.documentation_collection.add(
            documents=documentation,
            embeddings=self.generate_embedding(documentation),
            ids=id,
        )
        return id

    def get_training_data(self, **kwargs) -> pd.DataFrame:
        sql_data = self.sql_collection.get()

        df = pd.DataFrame()

        if sql_data is not None:
            # Extract the documents and ids
            documents = [json.loads(doc) for doc in sql_data["documents"]]
            ids = sql_data["ids"]

            # Create a DataFrame
            df_sql = pd.DataFrame(
                {
                    "id": ids,
                    "question": [doc["question"] for doc in documents],
                    "content": [doc["sql"] for doc in documents],
                }
            )

            df_sql["training_data_type"] = "sql"

            df = pd.concat([df, df_sql])

        ddl_data = self.ddl_collection.get()

        if ddl_data is not None:
            # Extract the documents and ids
            documents = [doc for doc in ddl_data["documents"]]
            ids = ddl_data["ids"]

            # Create a DataFrame
            df_ddl = pd.DataFrame(
                {
                    "id": ids,
                    "question": [None for doc in documents],
                    "content": [doc for doc in documents],
                }
            )

            df_ddl["training_data_type"] = "ddl"

            df = pd.concat([df, df_ddl])

        doc_data = self.documentation_collection.get()

        if doc_data is not None:
            # Extract the documents and ids
            documents = [doc for doc in doc_data["documents"]]
            ids = doc_data["ids"]

            # Create a DataFrame
            df_doc = pd.DataFrame(
                {
                    "id": ids,
                    "question": [None for doc in documents],
                    "content": [doc for doc in documents],
                }
            )

            df_doc["training_data_type"] = "documentation"

            df = pd.concat([df, df_doc])

        return df

    def remove_training_data(self, id: str, **kwargs) -> bool:
        if id.endswith("-sql"):
            self.sql_collection.delete(ids=id)
            return True
        elif id.endswith("-ddl"):
            self.ddl_collection.delete(ids=id)
            return True
        elif id.endswith("-doc"):
            self.documentation_collection.delete(ids=id)
            return True
        else:
            return False

    def remove_collection(self, collection_name: str) -> bool:
        """
        This function can reset the collection to empty state.

        Args:
            collection_name (str): sql or ddl or documentation

        Returns:
            bool: True if collection is deleted, False otherwise
        """
        if collection_name == "sql":
            self.chroma_client.delete_collection(name="sql")
            self.sql_collection = self.chroma_client.get_or_create_collection(
                name="sql", embedding_function=self.embedding_function
            )
            return True
        elif collection_name == "ddl":
            self.chroma_client.delete_collection(name="ddl")
            self.ddl_collection = self.chroma_client.get_or_create_collection(
                name="ddl", embedding_function=self.embedding_function
            )
            return True
        elif collection_name == "documentation":
            self.chroma_client.delete_collection(name="documentation")
            self.documentation_collection = self.chroma_client.get_or_create_collection(
                name="documentation", embedding_function=self.embedding_function
            )
            return True
        else:
            return False

    @staticmethod
    def _extract_documents(query_results) -> list:
        """
        Static method to extract the documents from the results of a query.

        Args:
            query_results (pd.DataFrame): The dataframe to use.

        Returns:
            List[str] or None: The extracted documents, or an empty list or
            single document if an error occurred.
        """
        if query_results is None:
            return []

        if "documents" in query_results:
            documents = query_results["documents"]

            if len(documents) == 1 and isinstance(documents[0], list):
                try:
                    documents = [json.loads(doc) for doc in documents[0]]
                except Exception as e:
                    return documents[0]

            return documents

    def get_similar_question_sql(self, question: str, **kwargs) -> list:
        return AzureAISearch._extract_documents(
            self.sql_collection.query(
                query_texts=[question],
                n_results=self.n_results_sql,
            )
        )

    def get_related_ddl(self, question: str, **kwargs) -> list:
        return AzureAISearch._extract_documents(
            self.ddl_collection.query(
                query_texts=[question],
                n_results=self.n_results_ddl,
            )
        )

    def get_related_documentation(self, question: str, **kwargs) -> list:
        return AzureAISearch._extract_documents(
            self.documentation_collection.query(
                query_texts=[question],
                n_results=self.n_results_documentation,
            )
        )