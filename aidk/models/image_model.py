from openai import OpenAI
from ..keys.keys_manager import load_key

class ImageModel:
    """
    A class to interact with AI image generation models.
    
    ImageModel provides an interface for generating images from text prompts using
    AI models. Currently supports OpenAI's DALL-E 3, with potential for expansion
    to other providers and models in the future.

    Examples
    --------
    Basic image generation:
    ```
    model = ImageModel(provider="openai", model="dall-e-3")
    response = model.generate("A beautiful garden with flowers")
    ```
    """

    def __init__(self, provider: str, model: str):
        """
        Initialize a new ImageModel instance.

        Parameters
        ----------
        provider : str
            Name of the provider (currently only "openai" is supported)
        model : str
            Name of the model (currently only "dall-e-3" is supported)

        Raises
        ------
        ValueError
            If an unsupported provider or model is specified
        """
        self.provider = provider
        self.model = model

        if provider.lower() != "openai":
            raise ValueError(f"Provider {provider} not supported")
        if model.lower() != "dall-e-3":
            raise ValueError(f"Model {model} not supported")
        
        load_key(provider)
        self._client = OpenAI()

    def generate(self, 
                prompt: str, 
                size: str = "1024x1024", 
                quality: str = "standard", 
                n: int = 1) -> dict:
        """
        Generate images from a text prompt.

        Parameters
        ----------
        prompt : str
            The text description of the image to generate
        size : str, optional
            The size of the generated image(s). Options:
            - "1024x1024" (default)
            - "1792x1024"
            - "1024x1792"
        quality : str, optional
            The quality of the generated image(s). Options:
            - "standard" (default)
            - "hd"
        n : int, optional
            Number of images to generate (default: 1)

        Returns
        -------
        dict
            OpenAI image generation response containing:
            - created: timestamp
            - data: list of generated images with URLs and other metadata
        """
        response = self._client.images.generate(
            model=self.model,
            prompt=prompt,
            size=size,
            quality=quality,
            n=n,
        )
        return response



