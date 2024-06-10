from vanna.openai import OpenAI_Chat
from vanna.chromadb import ChromaDB_VectorStore
from flask import Flask, jsonify, request
from config import DefaultConfig
from dotenv import load_dotenv, find_dotenv
import requests

load_dotenv(find_dotenv())

CONFIG = DefaultConfig()

class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config={'path':'chroma'})
        OpenAI_Chat.__init__(self, config=config)

print(CONFIG.OPENAI_APIKEY)
print(CONFIG.MSSQL_CONNECTION_STRING)

vn = MyVanna(config={'api_key': CONFIG.OPENAI_APIKEY, 'model': CONFIG.OPENAI_DEPLOYMENT})
vn.connect_to_mssql(odbc_conn_str=CONFIG.MSSQL_CONNECTION_STRING)

df = vn.get_training_data()
if df.empty:
    df_information_schema = vn.run_sql(CONFIG.MSSQL_BOOTSTRAP_QUERY)
    plan = vn.get_training_plan_generic(df_information_schema)
    vn.train(plan=plan)

response = requests.get(CONFIG.TRAINING_ENDPOINT)
if response.ok:
    json = response.json()

    questions = json.get('questions')
    if questions is not None:
        for item in questions:
            question = item.get('question')
            sql = item.get('sql')
            print(f"{question}={sql}")
            vn.train(question=question,sql=sql)

    sqls = json.get('sqls')
    if sqls is not None:
        for item in sqls:
            sql = item.get('sql')
            print(f"={sql}")
            vn.train(sql=sql)

    documentations = json.get('documentations')
    if documentations is not None:
        for item in documentations:
            documentation = item.get('documentation')
            print(f"{documentation}=")
            vn.train(documentation=documentation)

app = Flask(__name__, static_url_path='')

@app.route('/answer', methods=['POST'])
def answer():
    content = request.json
    question = content['question']
    try:
        sql = vn.generate_sql(question=question)
        df = vn.run_sql(sql=sql)
        csv = df.to_csv()
        return jsonify({"preview": df.head(10), "csv": csv, "sql": sql})
    except Exception as e:
        return jsonify({"error": True, "detail" : sql})

@app.route('/')
def root():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=8080)