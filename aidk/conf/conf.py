import os
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
import yaml
from yaml.parser import ParserError

class Conf:
    """
    A singleton class for managing configuration in a programmatic way.
    Configuration defined here overrides the configuration in ai.yaml file.

    Examples
    --------
    ```
    from aidk.conf import Conf
    
    # override the base model
    Conf()["base_model"] = {
        "provider": "openai",
        "model": "gpt-4o-mini"
    }
    
    # set prompts path programmatically
    Conf()["prompts_path"] = "prompts"
    ```
    """
    
    _instance = None
    _DEFAULT_CONFIG = {
        'keysfile_path': "providers.keys",
        'supported_files': {
            "text": ["txt", "py", "md"],
            "image": ["png", "jpg", "jpeg", "gif", "webp"]
        },
        'prompts_path': "",
        'default_prompt': {
            "rag": "Use also the following information to answer the question: ",
            "summary":"Here is the summary of the conversation so far:\n\n",
            "file":"Here is the content of the file:\n\n"
        },
        'observability': []
    }
    
    def __new__(cls):
        """
        Create or return the singleton instance.

        Returns
        -------
        Config
            The singleton instance
        """
        if cls._instance is None:
            cls._instance = super(Conf, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """
        Initialize the Config instance.
        
        Loads and validates the configuration file, setting up defaults
        and performing environment variable interpolation.
        """
        self._config_path = Path('ai.yaml')
        self._config = self._DEFAULT_CONFIG.copy()
        self._load_config()
        self._load_observability_keys()
    
    def _load_config(self) -> None:
        """
        Load configuration from the YAML file.
        
        Handles file reading, YAML parsing, environment variable interpolation,
        and configuration validation.

        Raises
        ------
        FileNotFoundError
            If the configuration file doesn't exist
        yaml.ParserError
            If the YAML syntax is invalid
        ValueError
            If the configuration structure is invalid
        """
        try:
            if self._config_path.exists():
                with open(self._config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                if file_config:
                    self._merge_config(self._config, file_config)            
            
        except ParserError as e:
            raise ValueError(f"Invalid YAML syntax in {self._config_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration: {e}")
    
    def _load_observability_keys(self) -> None:
        """
        Load observability environment variables from observability.keys file.
        
        Only loads keys for services that are present in the observability list
        in the configuration. Matches keys by checking if the first part of the
        key name matches any service in the observability list.
        """
        try:
            # Check if observability list exists and is not empty
            observability_services = self._config.get('observability', [])
            if not observability_services or not isinstance(observability_services, list):
                return
            
            # Load observability.keys file
            observability_keys_path = Path('observability.keys')
            if not observability_keys_path.exists():
                return
            
            with open(observability_keys_path, 'r') as f:
                lines = f.readlines()
            
            # Process each line in the keys file
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Parse key=value format
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Check if this key should be loaded based on observability services
                    if self._should_load_key(key, observability_services):
                        os.environ[key] = value
                        
        except Exception as e:
            # Don't raise error, just log it as observability is optional
            print(f"Warning: Failed to load observability keys: {e}")
    
    def _should_load_key(self, key: str, observability_services: List[str]) -> bool:
        """
        Check if a key should be loaded based on the observability services list.
        
        Parameters
        ----------
        key : str
            The environment variable key to check
        observability_services : List[str]
            List of observability services to load keys for
            
        Returns
        -------
        bool
            True if the key should be loaded, False otherwise
        """
        # Extract the first part of the key (before first underscore or dot)
        key_prefix = key.split('_')[0].split('.')[0].upper()
        
        # Check if any service in the list matches the key prefix
        for service in observability_services:
            if isinstance(service, str):
                service_upper = service.upper()
                if key_prefix == service_upper or key_prefix.startswith(service_upper):
                    return True
        
        return False
    
    def _merge_config(self, base: Dict, override: Dict) -> None:
        """
        Recursively merge override configuration into base configuration.

        Parameters
        ----------
        base : Dict
            The base configuration dictionary
        override : Dict
            The override configuration dictionary
        """
        for key, value in override.items():
            if (
                key in base and 
                isinstance(base[key], dict) and 
                isinstance(value, dict)
            ):
                self._merge_config(base[key], value)
            else:
                base[key] = value
        
        
    def __getitem__(self, key: str) -> Any:
        return self._config[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        self._config[key] = value