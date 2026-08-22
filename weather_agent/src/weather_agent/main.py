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

