import os
import sys
import pymongo
from dotenv import load_dotenv
from security.exception.exception import NetworkSecurityException
from security.logging.logger import logging

# Load environment variables
load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")

class NetworkDataRetriever:
    def __init__(self):
        try:
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def fetch_records(self, database, collection, query={}, limit=5):
        """
        Fetch selective records from MongoDB collection.

        Args:
            database (str): Database name
            collection (str): Collection name
            query (dict): MongoDB filter query (default: {})
            limit (int): Number of records to fetch (default: 5)

        Returns:
            list: Fetched records
        """
        try:
            db = self.mongo_client[database]
            coll = db[collection]
            results = list(coll.find(query).limit(limit))
            return results
        except Exception as e:
            raise NetworkSecurityException(e, sys)


if __name__ == '__main__':
    DATABASE = "tholathdb"
    COLLECTION = "NetworkData"

    retriever = NetworkDataRetriever()

    # Example: get records where 'having_IP_Address' == 1
    query = {"having_IP_Address": 1}

    try:
        records = retriever.fetch_records(DATABASE, COLLECTION, query=query, limit=3)
        for record in records:
            print(record)
    except Exception as e:
        print(f"❌ Failed to fetch records: {e}")
