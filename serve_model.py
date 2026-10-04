# server.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import uvicorn

app = FastAPI()
model_path = "./models/kaggle/gemma-4-31b"

print("A carregar o modelo de 31B para a memória unificada (aguarde)...")

# O compressed-tensors atua automaticamente aqui por baixo do capô
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",  # Usa automaticamente o chip M (MPS) ou CPU
    torch_dtype=torch.float16,
)
print("Modelo carregado! Servidor a ouvir na porta 8080.")


class ChatRequest(BaseModel):
    messages: list
    model: str = "default"


@app.post("/v1/chat/completions")
def chat(request: ChatRequest):
    # Formata a conversa para o padrão que o Gemma 4 entende
    prompt = tokenizer.apply_chat_template(
        request.messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Gera a resposta
    outputs = model.generate(**inputs, max_new_tokens=512)
    response_text = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
    )

    # Devolve no formato exato que o Google ADK espera
    return {"choices": [{"message": {"role": "assistant", "content": response_text}}]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
