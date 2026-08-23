import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL = os.getenv("MODEL")

if not MODEL:
    raise ValueError("MODEL environment variable is not set")

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url=BASE_URL
)

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": "What is Apple's stock ticker and current price?"
        }
    ]
)