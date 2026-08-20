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
5. Do not provide private chain-of-thought. Provide only a concise reasoning summary.
6. Continue producing THINKING steps until the answer is ready.
7. Once the answer is ready, return FINAL_OUTPUT.
8. FINAL_OUTPUT must contain only the answer intended for the user.

Example:

User:
Roger has 5 tennis balls. He buys 2 cans containing 3 tennis balls each. How many tennis balls does he have?

Response 1:
{
    "content": "Roger starts with 5 tennis balls.",
    "step_type": "THINKING"
}

Response 2:
{
    "content": "The 2 cans contain 2 × 3 = 6 tennis balls.",
    "step_type": "THINKING"
}

Response 3:
{
    "content": "Roger now has 5 + 6 = 11 tennis balls.",
    "step_type": "FINAL_OUTPUT"
}

IMPORTANT:
-Return only ONE step per API call.
-Don't call any external APIs or tools. Only use the information provided in the user query.
-Don't return any explanations or reasoning outside of the JSON object.
"""


def llm_json_reply(messages: list[dict]) -> str:
    provider = select_provider()
    client = build_client(provider)

    kwargs = {
        "model": provider.model,
        "max_tokens": 1000,
        "messages": messages,
    }

    if provider.name == "Groq":
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "cot_step",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string"
                        },
                        "step_type": {
                            "type": "string",
                            "enum": [
                                "THINKING",
                                "FINAL_OUTPUT"
                            ]
                        }
                    },
                    "required": [
                        "content",
                        "step_type"
                    ],
                    "additionalProperties": False
                }
            }
        }

    result = client.chat.completions.create(**kwargs)

    return result.choices[0].message.content

class CoTStep(BaseModel):
    """Class to represent a single step in the chain of thought."""
    content: str
    step_type: Literal["THINKING", "FINAL_OUTPUT"]


def parse_cot_step(response: str) -> CoTStep | str:
    """Parse the response from the LLM into a CoTStep object."""
    try:
        parsed = json.loads(response)
        return CoTStep(**parsed)
    except (json.JSONDecodeError, TypeError) as e:
        return f"Error parsing response: {e}. Response was: {response}"


class CoTAssistant:
    """An assistant that uses chain of thought reasoning to solve user queries."""

    def __init__(self):
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.MAX_STEPS = 15

    def run(self, user_query: str) -> str:
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query}
        ]

        for _ in range(self.MAX_STEPS):
            llm_response = llm_json_reply(self.messages)

            print(f"RAW RESPONSE: {llm_response}\n")

            self.messages.append({
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

while True:
    user_query = input("👉 Enter your query (or type 'exit' to quit): ")
    if user_query.lower() == "exit":
        break
    result = assistant.run(user_query)
    print(f"🎯: {result}")
    print("="*50)