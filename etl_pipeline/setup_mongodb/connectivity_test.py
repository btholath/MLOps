import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ConfigurationError
from dotenv import load_dotenv

def test_mongodb_connection():
    """
    Tests connection to a MongoDB database using URI from environment variables.
    """
    load_dotenv()  # Load variables from .env file

    uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_DB", "test")  # Default to 'test' if not set

    if not uri:
        print("❌ MONGODB_URI is not set in environment or .env file.")
        return

    try:
        print(f"🔌 Connecting to MongoDB URI: {uri}")
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)  # 5s timeout
        db = client[db_name]

        # Ping the server to confirm connectivity
        server_info = client.server_info()
        print(f"✅ Connected to MongoDB version: {server_info.get('version')}")
        print(f"📦 Available collections in '{db_name}': {db.list_collection_names()}")

    except ConnectionFailure as cf:
        print(f"❌ Connection failed: {cf}")
    except ConfigurationError as ce:
        print(f"❌ Configuration error: {ce}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_mongodb_connection()
