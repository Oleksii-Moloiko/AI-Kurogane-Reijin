import ollama

def ask_llm(history: list, model: str = "gemma3") -> str:
    try:
        response = ollama.chat(
            model=model,
            messages=history
        )
        return response['message']['content']
    except Exception as e:
        print(f"Помилка звернення до Ollama: {e}")
        return "Вибачте, сталася помилка при обробці запиту."
    