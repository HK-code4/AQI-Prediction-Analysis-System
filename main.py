from pymongo import MongoClient

import os
mongo_uri = os.getenv("MONGO_URI")

client = MongoClient(mongo_uri)

db = client["aqi_db"]
raw_collection = db["raw_data"]
feature_collection = db["features"]

print("✅ Connected to MongoDB Atlas")





