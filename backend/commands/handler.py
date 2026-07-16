"""Terminal command handlers."""
import shutil
from pathlib import Path

from rich.prompt import Confirm
from backend.chat.session import ChatSession
from backend.config import settings
from backend.llm.types import LLMProvider
from backend.storage import ChatRepository
from backend.ui.console import console
from backend.ui.layout import print_banner, print_footer, print_header
from backend.ui.panels import (
    print_chat_list,
    print_error,
    print_help,
    print_history,
    print_model_list,
    print_system,
)
from backend.ui.prompt import Command, read_chat_choice, read_model_choice
from backend.ui.sources import print_rag_sources
from backend.ui.documents import (
    print_document_removed,
    print_documents,
    print_indexed_document,
)
from backend.utils.logger import get_logger
from backend.rag.errors import (
    DocumentError,
    DuplicateDocumentError,
)
from backend.rag.indexer import DocumentIndexer


logger = get_logger(__name__)


def select_model(current_model: str, provider: LLMProvider) -> str:
    """Allow the user to select a model exposed by the active provider."""
    try:
        models = provider.list_models()
    except Exception as error:
        logger.exception("Failed to retrieve models")
        print_error(f"Не вдалося отримати список моделей: {error}")
        return current_model

    if not models:
        print_error("Доступні моделі не знайдено.")
        return current_model

    print_model_list(models=models, current_model=current_model)
    try:
        choice = read_model_choice()
    except (EOFError, KeyboardInterrupt):
        print_system("Зміну моделі скасовано.")
        return current_model

    if not choice:
        print_system("Зміну моделі скасовано.")
        return current_model

    selected_model = resolve_model_choice(choice=choice, models=models)
    if selected_model is None:
        logger.warning("Invalid model selection: choice=%s", choice)
        print_error(f"Не вдалося знайти модель за значенням: {choice}")
        return current_model

    if selected_model == current_model:
        print_system(f"Модель {selected_model} уже активна.")
        return current_model

    print_system(f"Активну модель змінено: {current_model} → {selected_model}")
    return selected_model


def resolve_model_choice(choice: str, models: list[str]) -> str | None:
    if choice.isdigit():
        index = int(choice) - 1
        return models[index] if 0 <= index < len(models) else None
    normalized = choice.casefold()
    return next((model for model in models if model.casefold() == normalized), None)


def handle_command(
    command: Command,
    session: ChatSession,
    provider: LLMProvider,
    repository: ChatRepository,
    *,
    arguments: str = "",
    document_indexer: DocumentIndexer | None = None,
) -> ChatSession:
    
    logger.info("Command received: command=%s", command.value)
    if command is Command.CLEAR:
        session.reset(reset_title=True)
        redraw(session)
        print_system("Історію поточного чату очищено.")
    elif command is Command.HISTORY:
        print_history(session.messages)
    elif command is Command.HELP:
        print_help()
    elif command is Command.MODEL:
        previous_model = session.model
        session.update_model(select_model(session.model, provider))
        if session.model != previous_model:
            redraw(session)
    elif command is Command.NEW:
        session.discard_if_empty()
        session = ChatSession(
            model=session.model,
            provider=session.provider,
            repository=repository,
            max_history=settings.max_history,
        )
        redraw(session)
        print_system(f"Створено новий чат: {session.chat_id}")
    elif command is Command.SESSIONS:
        print_chat_list(repository.list_chats())
    elif command is Command.RESUME:
        session = resume_chat(session, repository)
    elif command is Command.INDEX:
        handle_index(
            arguments=arguments,
            document_indexer=document_indexer,
        )

    elif command is Command.DOCUMENTS:
        handle_documents(
            arguments=arguments,
            repository=repository,
        )

    elif command is Command.REMOVE:
        handle_remove(
            arguments=arguments,
            repository=repository,
        )

    elif command is Command.SOURCES:
        handle_sources(session=session)
    
    else:
        print_error(f"Команда {command.value} не підтримується.")
    
    return session

def handle_sources(
    *,
    session: ChatSession,
) -> None:
    """Show RAG sources used for the latest assistant reply."""

    rag_context = session.last_rag_context

    if rag_context is None or rag_context.is_empty:
        print_system(
            "Джерела для останньої відповіді відсутні."
        )
        return

    print_rag_sources(rag_context)


def handle_index(
    *,
    arguments: str,
    document_indexer: DocumentIndexer | None,
) -> None:
    """Index a PDF or DOCX document."""

    if document_indexer is None:
        print_error(
            "RAG indexer не налаштований."
        )
        return

    document_path = arguments.strip()

    if not document_path:
        print_error(
            "Вкажи шлях до PDF або DOCX.\n"
            "Приклад: /index /Users/me/Documents/report.pdf"
        )
        return

    # Підтримка шляхів, взятих у лапки.
    if (
        len(document_path) >= 2
        and document_path[0] == document_path[-1]
        and document_path[0] in {"'", '"'}
    ):
        document_path = document_path[1:-1]

    try:
        result = document_indexer.index(
            document_path
        )

    except DuplicateDocumentError as error:
        print_system(str(error))
        return

    except DocumentError as error:
        logger.warning(
            "Document indexing failed: %s",
            error,
        )
        print_error(str(error))
        return

    except Exception as error:
        logger.exception(
            "Unexpected document indexing failure"
        )
        print_error(
            f"Неочікувана помилка індексації: {error}"
        )
        return

    print_indexed_document(result)

def handle_documents(
    *,
    arguments: str,
    repository: ChatRepository,
) -> None:
    """Print indexed documents."""

    limit = 100
    raw_limit = arguments.strip()

    if raw_limit:
        try:
            limit = int(raw_limit)
        except ValueError:
            print_error(
                "Ліміт має бути цілим числом.\n"
                "Приклад: /documents 20"
            )
            return

    if limit <= 0:
        print_error(
            "Ліміт має бути більшим за нуль."
        )
        return

    limit = min(limit, 500)

    documents = repository.list_documents(
        limit=limit
    )

    chunk_counts = {
        document.id: (
            repository.count_document_chunks(
                document.id
            )
        )
        for document in documents
    }

    print_documents(
        documents,
        chunk_counts=chunk_counts,
    )

def handle_remove(
    *,
    arguments: str,
    repository: ChatRepository,
) -> None:
    """Delete document metadata, chunks and the workspace copy."""

    document_id = arguments.strip()

    if not document_id:
        print_error(
            "Вкажи ID документа.\n"
            "Приклад: /remove a12b34c56d78"
        )
        return

    document = repository.get_document(
        document_id
    )

    if document is None:
        print_error(
            f"Документ з ID '{document_id}' не знайдено."
        )
        return

    confirmed = Confirm.ask(
        (
            f"Видалити документ "
            f"[bold]{document.filename}[/bold] "
            f"[dim]({document.id})[/dim]?"
        ),
        default=False,
        console=console,
    )

    if not confirmed:
        print_system(
            "Видалення скасовано."
        )
        return

    stored_path = Path(
        document.source_path
    )

    # Indexer зберігає файл у:
    # data/documents/<document-id>/<filename>
    document_directory = stored_path.parent

    deleted = repository.delete_document(
        document.id
    )

    if not deleted:
        print_error(
            "Не вдалося видалити запис документа."
        )
        return

    if document_directory.exists():
        try:
            shutil.rmtree(
                document_directory
            )
        except OSError as error:
            logger.exception(
                "Failed to remove document directory: %s",
                document_directory,
            )

            print_error(
                "Запис із бази видалено, але не вдалося "
                f"видалити локальні файли: {error}"
            )
            return

    print_document_removed(
        filename=document.filename,
        document_id=document.id,
    )

def resume_chat(current: ChatSession, repository: ChatRepository) -> ChatSession:
    chats = [chat for chat in repository.list_chats() if chat.id != current.chat_id or not current.is_empty]
    print_chat_list(chats)
    if not chats:
        return current
    try:
        choice = read_chat_choice()
    except (EOFError, KeyboardInterrupt):
        print_system("Відновлення чату скасовано.")
        return current
    if not choice:
        return current
    selected_id = choice
    if choice.isdigit():
        index = int(choice) - 1
        if not 0 <= index < len(chats):
            print_error("Невірний номер чату.")
            return current
        selected_id = chats[index].id
    current.discard_if_empty()
    try:
        restored = ChatSession(
            model=current.model,
            provider=current.provider,
            repository=repository,
            max_history=current.max_history,
            chat_id=selected_id,
        )
    except ValueError:
        print_error(f"Чат '{selected_id}' не знайдено.")
        return current
    redraw(restored)
    print_system(f"Відновлено чат: {restored.title} ({restored.chat_id})")
    print_history(restored.messages)
    return restored


def redraw(session: ChatSession) -> None:
    console.clear()
    print_banner()
    print_header(
        model_name=session.model,
        memory_enabled=settings.memory_enabled,
        workspace=settings.workspace_path,
        provider_name=session.provider,
        chat_id=session.chat_id,
        chat_title=session.title,
    )
    print_footer()
