import tiktoken
from typing import Union, List, Dict

class TokenCounter:
    _instance = None
    _encoders: Dict[str, tiktoken.Encoding] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TokenCounter, cls).__new__(cls)
        return cls._instance

    def _get_encoder(self, model: str) -> tiktoken.Encoding:
        """Get or create an encoder for the specified model."""
        if model not in self._encoders:
            try:
                # Map model names to their corresponding encodings
                if "gpt" in model.lower():
                    self._encoders[model] = tiktoken.encoding_for_model(model)
                elif "claude" in model.lower():
                    self._encoders[model] = tiktoken.get_encoding("cl100k_base")
                elif "llama" in model.lower():
                    self._encoders[model] = tiktoken.get_encoding("cl100k_base")
                else:
                    # Default to cl100k_base for unknown models
                    self._encoders[model] = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                print(f"Warning: Could not get specific encoder for {model}, using cl100k_base. Error: {e}")
                self._encoders[model] = tiktoken.get_encoding("cl100k_base")
        
        return self._encoders[model]


    def count(self, model: str, input_text: Union[str, List[str]], output_text: Union[str, List[str]]) -> Dict[str, int]:

        encoder = self._get_encoder(model)
        
        if isinstance(input_text, list):
            input_text = "\n".join(input_text)
        if isinstance(output_text, list):
            output_text = "\n".join(output_text)

        input_tokens = len(encoder.encode(input_text))
        output_tokens = len(encoder.encode(output_text))

        return {"input_tokens":input_tokens, "output_tokens":output_tokens, "total_tokens":input_tokens + output_tokens}




if __name__ == "__main__":
    input_text = "Hello, how are you today?"
    output_text = "I'm doing well!"
    model = "gpt-4o-mini"
    print(TokenCounter().count_tokens(model, input_text, output_text))
    