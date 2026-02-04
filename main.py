from pymongo import MongoClient

#import os
#mongo_uri = os.getenv("MONGO_URI")

#client = MongoClient(mongo_uri)

MONGO_URI = "mongodb+srv://kazmihiba22_db_user:OMKVxwdLjXIHYDyQ@cluster0.a2nawnq.mongodb.net/?appName=Cluster0"

client = MongoClient(MONGO_URI)

db = client["aqi_db"]
raw_collection = db["raw_data"]
feature_collection = db["features"]

print("✅ Connected to MongoDB Atlas")




