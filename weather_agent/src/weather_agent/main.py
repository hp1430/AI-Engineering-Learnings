import os
from openai import OpenAI
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

weather_schema = {
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