#FROM python:3.12.4-bookworm
FROM laudio/pyodbc:3.0.0

WORKDIR /opt/app

RUN pip3 install --upgrade pip
RUN pip install vanna[openai,chromadb]

COPY config.py .
COPY server.py .
COPY index.html .

EXPOSE 8080

CMD [ "python", "./server.py" ]
