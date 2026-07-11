import ollama

def ask_llm(history: list, model: str = "gemma3") -> str:
    response = ollama.chat(
        model=model,
        messages=history
    )
    return response['message']['content']
    

def ask_llm_stream(history: list, model: str ="gemma3"):
    stream = ollama.chat(
        model=model,
        messages=history,
        stream=True,
    )
    for part in stream:
        content = part.get("message", {}).get("content", "")
        if content:
            yield content