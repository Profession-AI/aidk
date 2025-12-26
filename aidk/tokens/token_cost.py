from typing import Dict
from .token_pricing import TOKEN_PRICING

class TokenCost:

    _TOKEN_COST_FILE = "token_pricing.json"

    def compute(self, provider: str, model: str, input_tokens: int, output_tokens: int) -> Dict[str, float]:
        """
        Compute token costs for input and output tokens.
        
        Args:
            provider: The provider name (e.g., 'openai', 'anthropic')
            model: The model name (e.g., 'gpt-4', 'claude-3')
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Dictionary containing input_cost, output_cost, and total_cost
            
        Raises:
            FileNotFoundError: If token_pricing.json is not found
            KeyError: If provider or model is not found in pricing file
            ValueError: If input_tokens or output_tokens are negative
            json.JSONDecodeError: If token_pricing.json is invalid JSON
        """
        # Validate input parameters
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("Token counts cannot be negative")


        # Validate provider exists
        if provider not in TOKEN_PRICING:
            raise KeyError(f"Provider '{provider}' not found in pricing file")

        # Validate model exists
        if model not in TOKEN_PRICING[provider]:
            raise KeyError(f"Model '{model}' not found for provider '{provider}'")

        # Validate pricing structure
        if not all(key in TOKEN_PRICING[provider][model] for key in ["input", "output"]):
            raise KeyError(f"Missing pricing information for model '{model}'")

        try:
            input_cost = round(input_tokens * TOKEN_PRICING[provider][model]["input"] / 1000, 8)
            output_cost = round(output_tokens * TOKEN_PRICING[provider][model]["output"] / 1000, 8)
            total_cost = round(input_cost + output_cost, 8)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Error calculating costs: {str(e)}")

        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost
        }
    