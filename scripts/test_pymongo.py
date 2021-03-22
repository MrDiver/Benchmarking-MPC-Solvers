from pymongo import MongoClient

client = MongoClient("192.168.0.101", 27017)

db = client.testremote

experiment_collection = db.experiments

exp = {"Name": "Experiment No.1", "config": {
    "T": 100, "K": 50, }, "states": [(0, 0, 0, 0)], "costs": [0]}

expid = experiment_collection.insert_one(exp).inserted_id
print(expid)
