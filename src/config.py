import os

class DefaultConfig:
    PORT = 3978
    APP_ID = os.environ.get("MicrosoftAppId", "")
    APP_PASSWORD = os.environ.get("MicrosoftAppPassword", "")
    APP_TYPE = os.environ.get("MicrosoftAppType", "MultiTenant")
    APP_TENANTID = os.environ.get("MicrosoftAppTenantId", "")
    OPENAI_APIKEY = os.environ.get("OPENAI_APIKEY", "")
    OPENAI_DEPLOYMENT = os.environ.get("OPENAI_DEPLOYMENT", "")
    MSSQL_CONNECTION_STRING = os.environ.get("MSSQL_CONNECTION_STRING", "")