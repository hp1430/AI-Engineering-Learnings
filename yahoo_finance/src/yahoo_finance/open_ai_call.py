import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL = os.getenv("MODEL")

if not MODEL:
    raise ValueError("MODEL environment variable is not set")

model = MODEL

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url=BASE_URL
)

def llm_reply(messages, TOOL_DEFINITION):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOL_DEFINITION
    )
    return response