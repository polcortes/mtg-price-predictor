from pymongo import MongoClient
from tqdm import tqdm
from datetime import datetime
import json


DB_NAME = "mtg_data"
COLLECTION_NAME = "card_price"

print("Initializing MongoDB client...", end=" ")

client = MongoClient("localhost", 27017)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

print("✅")
print("Retrieving the new data...", end=" ")

with open('./AllPrices.json', mode='rb') as file:
    data = list(json.load(file)['data'].items())
    
    print("✅")

    print("Inserting MTG cards into the database...")
    with client.start_session() as mongo_session:
        try:
            with mongo_session.start_transaction():
                n_documents_old_collection = collection.count_documents({})
                if n_documents_old_collection > 0:
                    collection.delete_many({})

                for idx in tqdm(range(len(data) - 1)):
                    card_id, values = data[idx]
                    
                    if not "paper" in values: 
                        continue
                    
                    retail = values['paper']['cardmarket']['retail']

                    if "foil" in retail:
                        foil_items = retail['foil']
                        collection.insert_one({"cardId": card_id, "isFoil": True, "datePriceMap": foil_items})
                    
                    if "normal" in retail:
                        normal_items = values['paper']['cardmarket']['retail']['normal']
                        collection.insert_one({"cardId": card_id, "isFoil": False, "datePriceMap": normal_items})
        except Exception:
            print("An error has occurred, rolling back the changes...")
