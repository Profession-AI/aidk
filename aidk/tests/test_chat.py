import pytest
from aidk.chat import Chat
from aidk.chat.history import JSONHistory
from aidk.chat.chat import ChatResponse
from aidk.models import Model
from aidk.models._response_processor import ModelStreamChunk, ModelStreamTail

TEST_MODEL = "gpt-4.1-nano"
TEST_SYSTEM_PROMPT = "You are a calculator, return only the number without any any other text"

"""
@pytest.fixture(autouse=True)
def cleanup_histories():
    yield
    # Delete JSON history folder
    json_history_path = Path("histories")
    if json_history_path.exists():
        shutil.rmtree(json_history_path)
    
    # Delete SQLite history folder
    sqlite_history_path = Path("sqlite_histories")
    if sqlite_history_path.exists():
        shutil.rmtree(sqlite_history_path)
"""

@pytest.fixture
def dict_chat():
    """Create a new chat with JSON history."""
    model = Model(provider="openai", model=TEST_MODEL)
    return Chat(model=model, system_prompt=TEST_SYSTEM_PROMPT, history="dict")


@pytest.fixture
def json_chat():
    """Create a new chat with JSON history."""
    model = Model(provider="openai", model=TEST_MODEL)
    return Chat(model=model, system_prompt=TEST_SYSTEM_PROMPT, history="json")

@pytest.fixture
def sqlite_chat():
    """Create a new chat with SQLite history."""
    model = Model(provider="openai", model=TEST_MODEL)
    return Chat(model=model, system_prompt=TEST_SYSTEM_PROMPT, history="sqlite")

def test_chat_initialization():
    """Test that chat is properly initialized with required attributes."""
    model = Model(provider="openai", model=TEST_MODEL)
    chat = Chat(model=model, system_prompt=TEST_SYSTEM_PROMPT)
    assert chat.chat_id is not None
    assert chat._history is not None

def test_history_basic_operations(dict_chat):
    """Test basic operations with dict history."""
    # Test initial calculation
    response = dict_chat.send("Io sono Giuseppe")
    response = dict_chat.send("dici il mio nome, solo il mio nome")
    assert response.response == "Giuseppe"

def test_json_history_basic_operations(json_chat):
    """Test basic operations with JSON history."""
    # Test initial calculation
    response = json_chat.send("Io sono Giuseppe")
    response = json_chat.send("dici il mio nome, solo il mio nome")
    assert response.response == "Giuseppe"

def test_json_history_persistence(json_chat):
    """Test that JSON history persists between chat instances."""
    # First chat instance
    response = json_chat.send("Ciao")
    response = json_chat.send("Io sono Giuseppe")
    
    # New chat instance with same history
    model = Model(provider="openai", model=TEST_MODEL)
    new_chat = Chat(model=model, history="json", chat_id=json_chat.chat_id)
    response = new_chat.send("dici il mio nome, solo il mio nome")
    assert response.response == "Giuseppe"

def test_sqlite_history_basic_operations(sqlite_chat):
    """Test basic operations with SQLite history."""
    # Test initial calculation
    response = sqlite_chat.send("Io sono Giuseppe")
    response = sqlite_chat.send("dici il mio nome, solo il mio nome")
    assert response.response == "Giuseppe"

def test_sqlite_history_persistence(sqlite_chat):
    """Test that SQLite history persists between chat instances."""
    # First chat instance
    sqlite_chat.send("Io sono Giuseppe")
    sqlite_chat.send("dici il mio nome, solo il mio nome")


def test_summarizer():
    """Test that the summarizer works."""
    model = Model(provider="openai", model=TEST_MODEL)
    history = JSONHistory(
        path="histories",
        summarizer_provider="openai",
        summarizer_model="gpt-4.1-nano",
        summarizer_max_tokens=200)
    new_chat = Chat(model=model,history=history)
    new_chat.send("Mi chiamo Giuseppe")
    response = new_chat.send("Come mi chiamo? Ritorna solo il mio nome e niente altro.")
    assert response.response == "Giuseppe"

def test_history_last_n():
    """Test that the history last_n parameter works."""

    last_n = 2
    model = Model(provider="openai", model=TEST_MODEL)
    history = JSONHistory(path="histories", last_n=last_n)
    json_chat = Chat(model=model, history=history)

    json_chat.send("Io sono Giuseppe, non ripetere il mio nome a meno che non te lo chieda")
    json_chat.send("Vivo in Italia")
    json_chat.send("Sono un ingegnere")
    json_chat.send("Ho un gatto di nome Nigma")
    response = json_chat.send("Come mi chiamo ? Ritorna il nome e niente altro.", return_history=True)
    assert len(response.history) == last_n*2+1  # last_n exchanges + system prompt
    assert response.response != "Giuseppe"

def test_send_stream_basic(json_chat):
    """Basic test for the streaming send interface."""
    import asyncio
    from aidk.chat import ChatStreamHead, ChatStreamChunk, ChatStreamTail

    async def _run():
        response = json_chat.send_stream("Raccontami una breve storia")
        head = True
        async for chunk in response:
            if head:
                assert isinstance(chunk, ChatStreamHead)
                head = False
            else:
                assert isinstance(chunk, (ChatStreamChunk, ChatStreamTail))

    asyncio.run(_run())

def test_send_async_basic(json_chat):
    """Basic test for the async non-streaming send_async interface."""
    import asyncio

    chat_response = asyncio.run(json_chat.send_async("Mi chiamo Giuseppe"))
    chat_response = asyncio.run(json_chat.send_async("Come mi chiamo? Ritorna solo il mio nome e niente altro."))

    assert chat_response.response == "Giuseppe"

def test_dict_history_summary():
    """Test get_summary and save_summary with DictHistory."""
    from aidk.chat.history import DictHistory
    
    history = DictHistory()
    chat_id = history.generate_chat_id()
    
    # Initially empty summary
    summary = history.get_summary(chat_id)
    assert summary.content == ""
    assert summary.role == "system"
    
    # Save a summary
    test_summary = "Conversation about Giuseppe"
    history.save_summary(chat_id, test_summary)
    
    # Retrieve the summary
    summary = history.get_summary(chat_id)
    assert summary.content == test_summary

def test_json_history_summary():
    """Test get_summary and save_summary with JSONHistory."""
    history = JSONHistory(path="histories_test")
    chat_id = history.generate_chat_id()
    
    # Initially empty summary
    summary = history.get_summary(chat_id)
    assert summary.content == ""
    assert summary.role == "system"
    
    # Save a summary
    test_summary = "Conversation about Giuseppe"
    history.save_summary(chat_id, test_summary)
    
    # Retrieve the summary
    summary = history.get_summary(chat_id)
    assert summary.content == test_summary

def test_sqlite_history_summary():
    """Test get_summary and save_summary with SQLiteHistory."""
    from aidk.chat.history import SQLiteHistory
    
    history = SQLiteHistory(path="sqlite_histories_test/test.db")
    chat_id = history.generate_chat_id()
    
    # Initially empty summary
    summary = history.get_summary(chat_id)
    assert summary.content == ""
    assert summary.role == "system"
    
    # Save a summary
    test_summary = "Conversation about Giuseppe"
    history.save_summary(chat_id, test_summary)
    
    # Retrieve the summary
    summary = history.get_summary(chat_id)
    assert summary.content == test_summary

def test_dict_history_clear():
    """Test clear method with DictHistory."""
    from aidk.chat.history import DictHistory
    
    history = DictHistory()
    chat_id = history.generate_chat_id()
    
    # Add some messages
    from aidk.chat.history.models import Message
    messages = [Message(content="Hello", role="user")]
    history.save_message(chat_id, messages)
    
    # Save a summary
    history.save_summary(chat_id, "Test summary")
    
    # Verify data exists
    assert len(history.get_messages(chat_id)) > 0
    assert history.get_summary(chat_id).content == "Test summary"
    
    # Clear the chat
    history.clear(chat_id)
    
    # Verify data is cleared
    assert len(history.get_messages(chat_id)) == 0
    assert history.get_summary(chat_id).content == ""

def test_json_history_clear():
    """Test clear method with JSONHistory."""
    history = JSONHistory(path="histories_test")
    chat_id = history.generate_chat_id()
    
    # Add some messages
    from aidk.chat.history.models import Message
    messages = [Message(content="Hello", role="user")]
    history.save_message(chat_id, messages)
    
    # Save a summary
    history.save_summary(chat_id, "Test summary")
    
    # Verify data exists
    assert len(history.get_messages(chat_id)) > 0
    assert history.get_summary(chat_id).content == "Test summary"
    
    # Clear the chat
    history.clear(chat_id)
    
    # Verify data is cleared
    assert len(history.get_messages(chat_id)) == 0
    assert history.get_summary(chat_id).content == ""

def test_sqlite_history_clear():
    """Test clear method with SQLiteHistory."""
    from aidk.chat.history import SQLiteHistory
    
    history = SQLiteHistory(path="sqlite_histories_test/test.db")
    chat_id = history.generate_chat_id()
    
    # Add some messages
    from aidk.chat.history.models import Message
    messages = [Message(content="Hello", role="user")]
    history.save_message(chat_id, messages)
    
    # Save a summary
    history.save_summary(chat_id, "Test summary")
    
    # Verify data exists
    assert len(history.get_messages(chat_id)) > 0
    assert history.get_summary(chat_id).content == "Test summary"
    
    # Clear the chat
    history.clear(chat_id)
    
    # Verify data is cleared
    assert len(history.get_messages(chat_id)) == 0
    assert history.get_summary(chat_id).content == ""