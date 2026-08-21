import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from dataclasses import dataclass
from pydantic import BaseModel
from typing import Literal

load_dotenv()

@dataclass(frozen=True)
class Provider:
    """Provider class to return the available provider based on the environment variable."""
    name: str
    env_var: str
    base_url: str
    model: str

PROVIDERS = [
    Provider(
        name="OpenAI",
        env_var="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini"
    ),
    Provider(
        name="Gemini",
        env_var="GEMINI_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        model="gemini-3.5-flash"
    ),
    Provider(
        name="Open Router",
        env_var="OPEN_ROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        model="openai/gpt-4o"
    ),
    Provider(
        name="Groq",
        env_var="GROQ_API_KEY",
        base_url="https://api.groq.com/openai/v1",
        model="openai/gpt-oss-20b"
    )
]

def select_provider() -> Provider:
    """Select the provider based on the environment variable."""
    for provider in PROVIDERS:
        if os.getenv(provider.env_var):
            return provider
    raise ValueError("No valid provider found. Please set the appropriate environment variable.")

def build_client(provider: Provider) -> OpenAI:
    """Build the OpenAI client based on the selected provider."""
    api_key = os.getenv(provider.env_var)
    if not api_key:
        raise ValueError(f"API key for {provider.name} is not set in environment variables.")
    
    return OpenAI(
        api_key=api_key,
        base_url=provider.base_url
    )

SYSTEM_PROMPT = """
You are an assistant that solves user queries iteratively.
Each API call must return exactly ONE JSON object:
{
    "content": "string",
    "step_type": "THINKING" | "FINAL_OUTPUT"
}
Rules:
1. Return ONLY the JSON object.
2. Never return markdown.
3. Never return multiple JSON objects.
4. Each THINKING response must contain ONE concise reasoning step.
5. Continue producing THINKING steps until the answer is ready.
6. Once the answer is ready, return FINAL_OUTPUT.
7. FINAL_OUTPUT must contain only the answer intended for the user.
"""

def llm_json_reply(messages: list[dict]) -> str:
    provider = select_provider()
    client = build_client(provider)

    kwargs = {
        "model": provider.model,
        "max_tokens": 1000,
        "messages": messages,
    }

    result = client.chat.completions.create(**kwargs)
    message = result.choices[0].message
    
    # FIX: Handle cases where the model returns None content (e.g., safety/empty stops)
    if message.content is None:
        raise ValueError("The model returned an empty content payload (None).")
        
    return message.content

class CoTStep(BaseModel):
    content: str
    step_type: Literal["THINKING", "FINAL_OUTPUT"]

def parse_cot_step(response: str) -> CoTStep | str:
    """Parse the response from the LLM into a CoTStep object, stripping accidental markdown code blocks if present."""
    cleaned_response = response.strip()
    # Fallback cleanup if the model wraps JSON in markdown blocks despite rules
    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response[7:]
    if cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[:-3]
    cleaned_response = cleaned_response.strip()

    try:
        parsed = json.loads(cleaned_response)
        return CoTStep(**parsed)
    except (json.JSONDecodeError, TypeError) as e:
        return f"Error parsing response: {e}. Response was: {response}"

class CoTAssistant:
    """An assistant that uses chain of thought reasoning to solve user queries."""

    def __init__(self):
        self.MAX_STEPS = 15

    def run(self, user_query: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query}
        ]

        for _ in range(self.MAX_STEPS):
            try:
                llm_response = llm_json_reply(messages)
            except Exception as e:
                print(f"❌ API Error: {e}\n")
                return f"Error during generation: {e}"

            # Append assistant response to history
            messages.append({
                "role": "assistant",
                "content": llm_response
            })

            parsed = parse_cot_step(llm_response)

            if isinstance(parsed, CoTStep):
                if parsed.step_type == "THINKING":
                    print(f"💡 {parsed.content}\n")
                    continue

                if parsed.step_type == "FINAL_OUTPUT":
                    print(f"🎯 {parsed.content}\n")
                    return parsed.content
            else:
                print(f"❌ {parsed}\n")
                return parsed

        return "❌ Maximum steps reached without a FINAL_OUTPUT."

assistant = CoTAssistant()

if __name__ == "__main__":
    while True:
        user_query = input("👉 Enter your query (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            break
        result = assistant.run(user_query)
        print(f"🎯: {result}")
        print("="*50)