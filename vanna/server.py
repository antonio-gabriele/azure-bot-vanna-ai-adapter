from vanna.openai import OpenAI_Chat
from vanna.chromadb import ChromaDB_VectorStore
from flask import Flask, jsonify, request
from config import DefaultConfig
from dotenv import load_dotenv

load_dotenv()

CONFIG = DefaultConfig()

class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config={'path':'chroma'})
        OpenAI_Chat.__init__(self, config=config)

print(CONFIG.MSSQL_CONNECTION_STRING)

vn = MyVanna(config={'api_key': CONFIG.OPENAI_APIKEY, 'model': CONFIG.OPENAI_DEPLOYMENT})
vn.connect_to_mssql(odbc_conn_str=CONFIG.MSSQL_CONNECTION_STRING)
df_information_schema = vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS")
plan = vn.get_training_plan_generic(df_information_schema)
plan
vn.train(plan=plan)

app = Flask(__name__, static_url_path='')

@app.route('/answer', methods=['POST'])
def answer():
    content = request.json
    question = content['question']
    try:
        sql = vn.generate_sql(question=question)
        df = vn.run_sql(sql=sql)
        csv = df.to_csv()
        return jsonify({"preview": df.head(10).to_json(orient='records'), "csv": csv, "sql": sql})
    except Exception as e:
        return jsonify({"error": True, "detail" : sql})

@app.route('/')
def root():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(debug=True)