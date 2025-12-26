from os import environ
import sys
import os
from typing import Dict, Optional

_KEY_EXT = "_API_KEY"

class KeyManager:

    """
    Key management module for AI service providers.

    This module handles the loading and validation of API keys for various AI service
    providers (like OpenAI, Anthropic, etc.). It manages key retrieval from environment
    variables and ensures proper key configuration before API usage.

    The module read keys from a providers.keys file in the root directory.
    The file is a simple text file with the following format:
    PROVIDER_NAME=API_KEY

    For example:
    OPENAI=sk-proj-ABCDE12345
    DEEPSEEK=sk-proj-FGHIJ67890

    Keys are automatically loaded when you use a model, so you don't need to use this module directly unless you want to load keys from a different file or disable key loading.
    """

    _instance = None
    _keys = None
    # Avoid loading configuration at import time; use default and optionally override on first use
    _key_file_path = "providers.keys"
    _is_enabled = True

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KeyManager, cls).__new__(cls)
        return cls._instance

    def set_key_file(self, key_file_path: str):
        """
        Set the path to the keys file.

        This method allows you to specify a custom path for the keys file.
        By default, the file is expected to be named providers.keys and located in the root directory of the project.

        Parameters
        ----------
        key_file_path : str
            The path to the keys file
        """
        self._key_file_path = key_file_path

    def enabled(self, enable: bool):
        
        """
        Enable or disable key loading.

        This method allows you to toggle the key loading mechanism on and off.
        When disabled, the KeyManager will not attempt to load keys from the
        providers.keys file.

        Parameters
        ----------
        enable : bool
            True to enable key loading, False to disable
        """
        self._is_enabled = enable

    def load_key(self, provider: str):
        
        """
        Load and validate API key for a specific provider.

        This function checks for the presence of the appropriate API key in
        environment variables and validates its format. If the key is not found
        or is invalid, it raises an exception.

        Parameters
        ----------
        provider : str
            The name of the provider (e.g., 'openai', 'anthropic')

        Raises
        ------
        ValueError
            If the API key is not found in environment variables
        ValueError
            If the API key format is invalid
        
        This method is automatically called when you use a model, so you don't really need to use this method directly.
        """
        
        if not self._is_enabled:
            return

        key_name = provider.upper() + _KEY_EXT

        if self._keys is None:
            # Optionally pick key file path from configuration lazily on first use
            try:
                from aidk.conf import Conf  # local import to avoid eager YAML load
                cfg_path = Conf()["keysfile_path"]
                if cfg_path:
                    self._key_file_path = cfg_path
            except Exception:
                # Fallback to default path if configuration is unavailable
                pass
            self._load_keys_from_file()            

        key = self._keys.get(key_name)
        if key is None:
            raise ValueError(f"Key for {provider} not found")
        
        environ[key_name] = key


    def _load_keys_from_file(self):

        try:
            with open(self._key_file_path, 'r') as f:
                self._keys = {}
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    if '=' in line:
                        provider, key = line.split('=', 1)
                        provider, key = provider.strip().upper(), key.strip().strip('"')
                        
                        if not provider.endswith(_KEY_EXT):
                            provider += _KEY_EXT
                            
                        self._keys[provider] = key
                    else:
                        print(f"Warning: Invalid line format in providers.keys: {line}")
        except FileNotFoundError:
            raise FileNotFoundError("providers.keys file not found. Please create it with your API keys.")



def load_key(provider: str):

    key_name = provider.upper() + _KEY_EXT

    if key_name in environ:
        return
    
    if 'google.colab' in sys.modules:
        _load_keys_from_colab(key_name)
    else:
        KeyManager().load_key(provider)


def _load_keys_from_colab(key_name: str):
    from google.colab import userdata
    environ[key_name] = userdata.get(key_name)



def check_key(provider: str) -> bool:
    """
    Check if an API key is available for a specific provider.

    Parameters
    ----------
    provider : str
        The name of the provider (e.g., 'openai', 'anthropic')

    Returns
    -------
    bool
        True if a key is available and valid, False otherwise

    Examples
    --------
    Check key availability:
        >>> if check_key("openai"):
        ...     print("OpenAI key is available")
        ... else:
        ...     print("OpenAI key is not available")
    """
    key = _get_key(provider)
    return bool(key and validate_key(provider, key))

def validate_key(provider: str, key: str) -> bool:
    """
    Validate the format of an API key for a specific provider.

    Parameters
    ----------
    provider : str
        The name of the provider (e.g., 'openai', 'anthropic')
    key : str
        The API key to validate

    Returns
    -------
    bool
        True if the key format is valid, False otherwise

    Examples
    --------
    Validate OpenAI key:
        >>> is_valid = validate_key("openai", "sk-...")
        >>> print(f"Key is {'valid' if is_valid else 'invalid'}")

    Validate Anthropic key:
        >>> is_valid = validate_key("anthropic", "sk-ant-...")
        >>> print(f"Key is {'valid' if is_valid else 'invalid'}")
    """
    if not key:
        return False
    
    validation_rules = {
        "openai": lambda k: k.startswith("sk-"),
        "anthropic": lambda k: k.startswith("sk-ant-"),
        # Add more provider validation rules as needed
    }
    
    validator = validation_rules.get(provider.lower())
    return bool(validator and validator(key))

def _get_key(provider: str) -> Optional[str]:
    """
    Get API key from environment variables for a specific provider.

    This is an internal function that handles the retrieval of API keys
    from environment variables using a standardized naming convention.

    Parameters
    ----------
    provider : str
        The name of the provider (e.g., 'openai', 'anthropic')

    Returns
    -------
    Optional[str]
        The API key if found, None otherwise

    Notes
    -----
    The function uses the following naming convention for environment variables:
    - OpenAI: OPENAI_API_KEY
    - Anthropic: ANTHROPIC_API_KEY
    - Other providers: PROVIDER_API_KEY (uppercase)
    """
    env_var = f"{provider.upper()}_API_KEY"
    return os.getenv(env_var)

# Optional: Provider-specific key validation patterns
KEY_PATTERNS: Dict[str, str] = {
    "openai": r"^sk-[a-zA-Z0-9]{32,}$",
    "anthropic": r"^sk-ant-[a-zA-Z0-9]{32,}$",
    # Add more patterns as needed
}

if __name__ == "__main__":
    load_key("openai")
    print(environ["OPENAI_API_KEY"])
    load_key("deepseek")
    print(environ["DEEPSEEK_API_KEY"])

