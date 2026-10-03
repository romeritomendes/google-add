import os
import asyncio
from google.adk.agents import LlmAgent

os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"

agent = LlmAgent(
    name="hello_agent",
    model="ollama/gemma2:2b",
    instruction="És um assistente muito sarcástico e direto. Responde sempre em português.",
)


async def chat_loop():
    print("--- Chat Iniciado (Escreva 'sair' para terminar) ---")

    context_session = {}

    while True:
        pergunta = await asyncio.to_thread(input, "\nVocê: ")
        if pergunta.strip().lower() == "sair":
            break

        try:
            print("Agent: ", end="", flush=True)

            async for chunk in agent.run(ctx=context_session, node_input=pergunta):
                print(chunk, end="", flush=True)

            print()
        except Exception as e:
            print(f"[Erro na execução]: {e}")


if __name__ == "__main__":
    asyncio.run(chat_loop())
