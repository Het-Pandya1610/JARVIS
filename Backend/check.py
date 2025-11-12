from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GroqAPIKey"))

models_response = client.models.list()

for model in models_response.data:
    # See what attributes or keys exist in the Model object
    print(model.__dict__)  # Shows internal dictionary of the object
