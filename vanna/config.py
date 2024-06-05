import os

class DefaultConfig:
    PORT = 3978
    OPENAI_APIKEY = os.environ.get("OPENAI_APIKEY", "")
    OPENAI_DEPLOYMENT = os.environ.get("OPENAI_DEPLOYMENT", "")
    MSSQL_CONNECTION_STRING = os.environ.get("MSSQL_CONNECTION_STRING", "")