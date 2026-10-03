import os
import logging
from google.adk.agents.llm_agent import Agent

logging.getLogger("LiteLLM").setLevel(logging.ERROR)

# os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"
os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
os.environ["OPENAI_API_KEY"] = "ollama"


def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city."""
    return {"status": "success", "city": city, "time": "10:30 AM"}


root_agent = Agent(
    model="openai/qwen3.8:latest",
    name="root_agent",
    description="Tells the current time in a specified city.",
    instruction="You are a helpful assistant that tells the current time in cities. Use the 'get_current_time' tool for this purpose.",
    tools=[get_current_time],
)
