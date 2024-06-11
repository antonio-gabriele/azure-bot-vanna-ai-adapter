FROM python:3.12.4-bookworm

RUN apt-get update
RUN apt-get install -y curl apt-utils apt-transport-https build-essential

RUN wget https://packages.microsoft.com/config/ubuntu/22.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb
RUN dpkg -i packages-microsoft-prod.deb
RUN rm packages-microsoft-prod.deb
RUN apt-get update
RUN ACCEPT_EULA=Y apt-get install -y msodbcsql18 unixodbc-dev

WORKDIR /opt/app

RUN pip3 install --upgrade pip
RUN pip install vanna[openai,chromadb]
RUN pip install pyodbc

COPY config.py .
COPY app.py .
COPY index.html .

EXPOSE 8080

ENTRYPOINT [ "python" ]

CMD ["app.py" ]
