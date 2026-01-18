from aidk.models import Model
from aidk.prompts import Prompt
from aidk.models._response_processor import ModelResponse, ModelUsage, Model as ModelInfo, ModelStreamHead, ModelStreamChunk, ModelStreamTail
import pytest
import asyncio

TEST_PROMPT = "This is a test"
TEST_PROMPT_FOLDER = "aidk/tests/prompts"
TEST_MODEL = "gpt-4.1-nano"

@pytest.fixture
def model():
    return Model(provider="openai", model=TEST_MODEL)


def assert_basic_response(resp, expected_type=None):
    assert isinstance(resp, ModelResponse)
    assert isinstance(resp.model, ModelInfo)
    assert hasattr(resp, "usage")
    if resp.usage is not None:
        assert isinstance(resp.usage, ModelUsage)
    if expected_type is not None:
        assert isinstance(resp.response, expected_type)


def test_base_model(model):
    resp = model.ask(TEST_PROMPT)
    assert_basic_response(resp)

def test_base_model_with_prompt(model):
    prompt = Prompt(prompt_id=f"{TEST_PROMPT_FOLDER}/base.prompt")
    resp = model.ask(prompt=prompt)
    assert_basic_response(resp)

def test_model_with_token_counting(model):
    response = model.ask(TEST_PROMPT)
    assert_basic_response(response)
    assert isinstance(response.usage, ModelUsage)

def test_model_with_cost_counting(model):
    response = model.ask(TEST_PROMPT)
    assert_basic_response(response)
    assert isinstance(response.usage, ModelUsage)
    assert response.usage.cost is not None

def test_model_with_prompt_variables(model):
    prompt = Prompt(
        prompt_id=f"{TEST_PROMPT_FOLDER}/with_variables.prompt",
        prompt_data={"country": "Italy"}
    )
    response = model.ask(prompt)
    assert_basic_response(response, expected_type=str)
    assert len(response.response) > 0
    assert "Rome" in response.response

def test_model_with_formatted_prompt(model):
    test_text = "The quick brown fox jumps over the lazy dog. This is a test text for summarization."
    prompt = Prompt(
        prompt_id=f"{TEST_PROMPT_FOLDER}/with_formatting.prompt",
        prompt_data={"text": test_text}
    )
    response = model.ask(prompt)
    assert_basic_response(response, expected_type=str)
    assert len(response.response) > 0
    assert "-" in response.response  # Check for bullet points

def test_model_with_type(model):
    prompt = Prompt(
        prompt="2+2=",
        response_type=int
    )
    response = model.ask(prompt)
    assert_basic_response(response, expected_type=int)
    assert response.response == 4


def test_model_with_prompt_type(model):
    prompt = Prompt(prompt_id=f"{TEST_PROMPT_FOLDER}/with_type.prompt")
    response = model.ask(prompt)
    print(response)
    assert_basic_response(response, expected_type=int)
    assert response.response == 4


def test_model_stream(model):
    """Ensure stream yields head, chunks, and a tail with usage."""
    async def _run():
        first = True
        async for chunk in model.ask_stream("This is a streaming test"):
            if first:
                assert isinstance(chunk, ModelStreamHead)
                first = False
            else:
                assert isinstance(chunk, (ModelStreamChunk, ModelStreamTail))
        tail = chunk
        assert isinstance(tail, ModelStreamTail)
        assert isinstance(tail.response, str)
        assert isinstance(tail.usage, ModelUsage)

    asyncio.run(_run())


def test_model_async(model):
    """Call the model's async implementation and assert a ModelResponse."""
    async def _run():
        resp = await model.ask_async("This is an async test")
        assert_basic_response(resp)
        assert hasattr(resp, "response")

    asyncio.run(_run())