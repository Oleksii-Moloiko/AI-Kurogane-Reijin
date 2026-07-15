# 🤖 Kuro — AI Kurogane Reijin

Kuro — це локальний термінальний помічник на базі штучного інтелекту з постійними чатами SQLite та підключаємими постачальниками LLM.

## MVP features

- Адаптивний багатий інтерфейс терміналу з Markdown та потоковим виводом
- Постачальник Ollama
- Додатковий постачальник, сумісний з OpenAI
- Постійні сеанси чату в SQLite
- Ідентифікатори чату та автоматичні заголовки
- Перелік та відновлення попередніх сеансів, включаючи безпосереднє відтворення історії
- Налаштоване контекстне вікно за допомогою `MAX_HISTORY`
- Виправлення помилок, коли Ollama, кінцева точка API або налаштована модель недоступні
- Автоматичне очищення ніколи не використовуваних порожніх сеансів
- Журнали файлів, що обертаються
- Базові модульні тести

## Requirements

- Python 3.11+
- Ollama для локальних налаштувань за замовчуванням або API, сумісний з OpenAI

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

For Ollama:

```bash
ollama serve
ollama pull gemma3
```

## Run

```bash
python backend/main.py
```

## Commands

| Command | Action |
|---|---|
| `/help` | Показати команди |
| `/history` | Показати історію поточного чату |
| `/clear` | Очистити поточний чат, скинути його назву та зберегти його ID |
| `/model` | Вибрати модель, надану активним постачальником |
| `/new` | Створити новий постійний чат |
| `/sessions` | Показати список збережених чатів |
| `/resume` | Відновити чат за номером або ID |
| `Ctrl+C` | Вихід |

## Configuration

The main settings live in `.env`:

```env
LLM_PROVIDER=ollama
LLM_MODEL=gemma3
MAX_HISTORY=20
DATABASE_PATH=data/kuro.db
```

`MAX_HISTORY` обмежує лише повідомлення, що надсилаються моделі. Повна розмова зберігається в SQLite.

### OpenAI-compatible provider

Цей режим працює з OpenAI та сумісними серверами, такими як LM Studio або vLLM:

```env
LLM_PROVIDER=openai
LLM_MODEL=your-model-name
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your-key
```

Для локального сервера без автентифікації залиште `OPENAI_API_KEY` порожнім та встановіть його базову URL-адресу `/v1`.

## Data

- SQLite database: `data/kuro.db`
- Logs: `data/logs/kuro.log`

Ці шляхи можна перевизначити у `.env`.

## Tests

```bash
pytest -q
```

## Architecture

```text
backend/
├── chat/       # стан сеансу та політика контексту
├── commands/   # команди терміналу
├── llm/        # контракт постачальника та реалізації
├── storage/    # репозиторій SQLite
├── ui/         # Багатий рендеринг терміналу
└── utils/      # логування
```

## Stabilization notes (v0.3.1)

- `/model` перемальовує заголовок після успішної зміни.
- `/new` зберігає поточного вибраного постачальника та модель.
- `/resume` перемальовує екран та друкує відновлену розмову.
- Кількість повідомлень сеансу не враховує системний запит.
- Порожні чати видаляються під час виходу, створення нового чату або відновлення іншого чату.
- Помилки з'єднання Ollama з `httpx` перетворюються на читабельні помилки постачальника.
- Налаштовані моделі Ollama перевіряються, включаючи псевдоніми `:latest`.