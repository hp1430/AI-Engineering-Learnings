import os
import json
from typing import cast
from openai import OpenAI
from openai.types.chat import ChatCompletionAssistantMessageParam, ChatCompletionMessage, ChatCompletionMessageFunctionToolCall, ChatCompletionMessageParam, ChatCompletionToolParam
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()

@dataclass(frozen=True)
class Provider:
    """
        Provider class to return available provider based on the
        environment variables.
    """
    name: str
    env_var: str
    base_url: str
    model: str

PROVIDERS = [
    Provider(
        name="Groq",
        env_var="GROQ_API_KEY",
        base_url="https://api.groq.com/openai/v1",
        model="openai/gpt-oss-20b"
    )
]

def select_provider() -> Provider:
    """Select the provider based on environment variables"""

    for provider in PROVIDERS:
        if os.getenv(provider.env_var):
            return provider

    raise ValueError("No valid provider found. Please add the valid api key in environment variables")

def build_client(provider: Provider) -> OpenAI:
    """Build the openai client based on the selected provider"""

    api_key = os.getenv(provider.env_var)

    if not api_key:
        raise ValueError(f"API Key for {provider.name} is not set in environment variables")

    return OpenAI(
        api_key=api_key,
        base_url=provider.base_url
    )

WEATHER_DATA = {
    "london": {"celsius": 22, "sky": "cloudy"},
    "new york": {"celsius": 28, "sky": "sunny"},
    "tokyo": {"celsius": 30, "sky": "humid"},
    "sydney": {"celsius": 16, "sky": "rainy"},
    "cairo": {"celsius": 36, "sky": "clear"},
    "paris": {"celsius": 24, "sky": "partly cloudy"},
    "reykjavik": {"celsius": 11, "sky": "windy"},
    "singapore": {"celsius": 31, "sky": "thunderstorm"},
    "mumbai": {"celsius": 32, "sky": "monsoon showers"},
    "toronto": {"celsius": 20, "sky": "clear"},
    "berlin": {"celsius": 19, "sky": "overcast"},
    "buenos aires": {"celsius": 14, "sky": "chilly"},
    "dubai": {"celsius": 41, "sky": "sunny"},
    "nairobi": {"celsius": 23, "sky": "breezy"},
    "seoul": {"celsius": 26, "sky": "humid"}
}

def lookup_weather(location: str) -> str:
    """Lookup the weather for a location"""

    record = WEATHER_DATA.get(location.lower())
    if record is None:
        return f"Sorry! I don't know the weather of {location}"

    return f"The weather of {location} is {record['celsius']}C and {record['sky']}."

weather_schema: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "lookup_weather",
        "description": "Look up the weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location to look up the weather for."
                }
            },
            "required": ["location"]
        }
    }
}

TOOLS = {
    "lookup_weather": lookup_weather
}

def ask_llm_with_tool(prompt: str, *, max_tokens: int = 400) -> str:
    """Call the LLM with a tool call"""

    provider = select_provider()
    client = build_client(provider)
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    while True:
        response = client.chat.completions.create(
            model=provider.model,
            max_tokens=max_tokens,
            messages=messages,
            tools=[weather_schema]
        )

        msg = response.choices[0].message

        if not msg.tool_calls:
            return msg.content or ""

        messages.append(cast(ChatCompletionAssistantMessageParam, {
            "role": "assistant",
            "content": msg.content,
            "tool_calls": msg.tool_calls
        }))

        # LLM is asking us to call a tool
        for tool_call in msg.tool_calls:
            if not isinstance(tool_call, ChatCompletionMessageFunctionToolCall):
                raise ValueError("Unsupported custom tool call")

            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments) # {"location": "london"}
            tool_call_id = tool_call.id

            if tool_name not in TOOLS:
                raise ValueError(f"Tool {tool_name} not found!")

            tool = TOOLS[tool_name] # actual function
            result = tool(**tool_args) # -> lookup_weather(location="London")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result
                }
            )

msg = ask_llm_with_tool("Compare the weathers of London and Mumbai")

print("final response from llm: ", msg)
   