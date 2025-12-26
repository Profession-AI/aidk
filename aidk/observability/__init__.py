"""
# AIDK Observability

AIDK provides built-in support for multiple observability and monitoring services to help you track, debug, and analyze your AI model interactions.

## Supported Services

### [Langfuse](https://www.langfuse.com/)
Open-source observability and analytics platform for LLM applications.

**Dependencies:**
```bash
pip install langfuse==2.59.7
```

### [Logfire](https://logfire.ai/)
Pydantic's observability platform for monitoring and debugging applications.

**Dependencies:**
```bash
pip install opentelemetry-api==1.25.0 opentelemetry-sdk==1.25.0 logfire
```

### [DeepEval](https://deeveval.com/)
Open-source evaluation framework for LLM applications.

### [LangSmith](https://www.langchain.com/langsmith)
LangChain's platform for debugging, testing, and monitoring LLM applications.

**Dependencies:**
```bash
pip install langsmith==0.1.11
```

⚠️ AIDK is currently based on litellm, so [other services supported by litellm](https://docs.litellm.ai/docs/observability/agentops_integration) could be used out-of-the-box.


## Setup

### 1. Configure Services

Add the `observability` field to your `ai.yaml` configuration file:

```yaml
observability: ["langfuse", "logfire"]
```

### 2. Add API Keys

Create an `observability.keys` file in your project root and add the required credentials:

```bash
# observability.keys
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LOGFIRE_TOKEN=your_logfire_token
```

### With Metadata

Pass custom metadata in the `metadata` parameter to track additional context with services that support it:

```python
# Synchronous request with metadata
response = model.ask(
    "Hello, how are you?", 
    metadata={
        "user_id": "12345",
        "session_id": "abc-def-ghi",
        "feature": "chat",
        "environment": "production"
    }
)
```
"""
