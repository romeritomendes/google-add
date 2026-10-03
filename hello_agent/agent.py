import os
import logging
from google.adk.agents.llm_agent import Agent

logging.getLogger("LiteLLM").setLevel(logging.ERROR)

os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"


def ler_arquivo(path: str) -> str:
    """Lê e devolve o conteúdo de um arquivo de texto local."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Erro ao ler o arquivo: {e}"


root_agent = Agent(
    model="ollama/gemma2:2b",
    name="hello_agent",
    description="A helpful assistant for user questions.",
    instruction="És um assistente muito sarcástico e direto. Responde sempre em português.",
    tools=[ler_arquivo],
)
