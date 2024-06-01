from vanna.openai import OpenAI_Chat
from vanna.chromadb import ChromaDB_VectorStore
from vanna.flask import VannaFlaskApp

from config import DefaultConfig

CONFIG = DefaultConfig()

class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)

vn = MyVanna(config={'api_key': CONFIG.OPENAI_APIKEY, 'model': CONFIG.OPENAI_DEPLOYMENT})
vn.connect_to_mssql(odbc_conn_str=CONFIG.MSSQL_CONNECTION_STRING)
df_information_schema = vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'BIDOC'")

plan = vn.get_training_plan_generic(df_information_schema)
vn.train(plan=plan)

app = VannaFlaskApp(vn)
app.run()