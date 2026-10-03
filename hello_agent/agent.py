import os
import logging
from google.adk.agents.llm_agent import Agent
from google.adk.workflow import Node, Workflow
from google.genai import types

logging.getLogger("LiteLLM").setLevel(logging.ERROR)

os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"
model = "ollama/gemma2:2b"


class FileReaderNode(Node):
    def run(self, ctx, node_input):
        if hasattr(node_input, "parts") and node_input.parts:
            filename = node_input.parts[0].text.strip()
        else:
            filename = str(node_input).strip()

        print(f"[system]: Tentando ler o arquivo '{filename}'...")

        try:
            with open(filename, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Erro ao ler o arquivo: {e}"


agent_resumer = Agent(
    model=model,
    name="agent_resumer",
    description="A helpful assistant for user questions.",
    instruction="És um assistente muito sarcástico e direto. Responde sempre em português.",
)


class MyWorkflow(Workflow):
    def run(self, ctx, node_input):
        reader = FileReaderNode(name="reader")
        txt = reader.run(ctx, node_input)

        print(f"[system]: A enviar dados para o Agent LLM...")

        txtContent = types.Content(
            role="user", parts=[types.Part(text=f"Analisa este texto: {txt}")]
        )

        return agent_resumer.run(ctx=ctx, node_input=txtContent)


root_agent = MyWorkflow(name="search_and_resume")
