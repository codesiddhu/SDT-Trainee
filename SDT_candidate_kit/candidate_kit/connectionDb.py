
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()
uri = os.getenv("MONGODB_URI")

client = MongoClient(uri)

try:
    client.admin.command("ping")
    print("MongoDB connected successfully!")

    db = client["my_database"]
    collection = db["students"]

    # Read all students
    students = collection.find()

    print("\nStudent Details:")

    for student in students:
        print(student)

except Exception as e:
    print("Error:", e)

finally:
    client.close()
