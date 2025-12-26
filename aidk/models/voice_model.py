from litellm import speech
from ..keys.keys_manager import load_key
import base64
import os

class VoiceModel:
    """
    A class for text-to-speech using various AI voice models.
    
    VoiceModel provides an interface for converting text to speech using different
    AI voice providers. Currently supports ElevenLabs and OpenAI voice models.
            
    Examples
    --------
    Basic usage with ElevenLabs:
    >>> model = VoiceModel(provider="elevenlabs", model="eleven_multilingual_v2", voice="21m00Tcm4TlvDq8ikWAM")
    >>> audio_file = model.speak("Hello, world!")
    
    Using different return types:
    >>> # Save as MP3 file
    >>> audio_file = model.speak("Hello, world!", return_type="output.mp3")
    
    >>> # Get as base64 string
    >>> audio_b64 = model.speak("Hello, world!", return_type="base64")
    
    >>> # Get as bytes
    >>> audio_bytes = model.speak("Hello, world!", return_type="bytes")
    
    Streaming audio generation:
    >>> async for audio_chunk in model.stream("Long text to convert to speech"):
    ...     print(f"Generated audio chunk: {audio_chunk}")
    """
    
    def __init__(self, provider: str, model: str, voice: str):
        """
        Initialize a new VoiceModel instance.
        
        Parameters
        ----------
        provider : str
            The voice provider to use (e.g., 'elevenlabs', 'openai')
        model : str
            The specific voice model to use
        voice : str
            The voice ID or voice name to use for speech generation
            
        Raises
        ------
        ImportError
            If the required provider library is not installed
        """
        load_key(provider)
        self._provider = provider
        self._model = model
        self._voice = voice
        
        if self._provider == "elevenlabs":
            try:
                from elevenlabs.client import ElevenLabs
            except ImportError:
                raise ImportError("elevenlabs is not installed. Please install it with 'pip install elevenlabs'")

            self._client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
        
    def speak(self, text: str, min_chars_per_sentence=100, return_type="audio.mp3"):
        """
        Convert text to speech and return the audio in the specified format.
        
        The method automatically splits long text into optimal chunks based on
        sentence boundaries to ensure high-quality audio generation.
        
        Parameters
        ----------
        text : str
            The text to convert to speech
        min_chars_per_sentence : int, optional
            Minimum number of characters per sentence group for optimal audio
            generation (default: 100)
        return_type : str, optional
            The format to return the audio in. Options:
            - File path ending with '.mp3' or '.wav' to save to file
            - 'base64' to return as base64 encoded string
            - 'bytes' to return as raw bytes (default: "audio.mp3")
            
        Returns
        -------
        str or bytes
            The generated audio in the specified format:
            - If return_type is a file path: returns the file path
            - If return_type is 'base64': returns base64 encoded string
            - If return_type is 'bytes': returns raw audio bytes
            
        Raises
        ------
        ValueError
            If return_type is not a valid option
            
        Examples
        --------
        >>> model = VoiceModel(provider="elevenlabs", model="eleven_multilingual_v2", voice="21m00Tcm4TlvDq8ikWAM")
        >>> # Save to file
        >>> audio_file = model.speak("Hello, world!", return_type="output.mp3")
        >>> # Get as base64
        >>> audio_b64 = model.speak("Hello, world!", return_type="base64")
        >>> # Get as bytes
        >>> audio_bytes = model.speak("Hello, world!", return_type="bytes")
        """
        audio_groups = self._generate_audio_groups(text, min_chars_per_sentence)
        audio_chunks = []
        
        for group in audio_groups:
            
            response = self._generate(group)
            
            audio_chunks.append(response)
        
        if return_type.endswith(".mp3") or return_type.endswith(".wav"):
            combined_audio = self._combine_bytes_chunks(audio_chunks)
            with open(return_type, 'wb') as f:
                f.write(combined_audio)
            return return_type
        elif return_type == "base64":
            combined_audio = self._combine_bytes_chunks(audio_chunks)
            return base64.b64encode(combined_audio).decode('utf-8')
        elif return_type == "bytes":
            return self._combine_bytes_chunks(audio_chunks)
        else:
            raise ValueError(f"Invalid return type: {return_type}")
    
    def _combine_bytes_chunks(self, audio_chunks: list) -> bytes:
        """
        Combine audio chunks into a single bytes object.
        
        Parameters
        ----------
        audio_chunks : list
            List of audio byte chunks to combine
            
        Returns
        -------
        bytes
            Combined audio data as bytes
        """
        combined = b""
        for chunk in audio_chunks:
            combined += chunk
        return combined
    
    def _generate_audio_groups(self, text: str, min_chars_per_sentence: int):
        """
        Generate optimized sentence groups for audio generation.
        
        Splits text into sentence groups that meet the minimum character requirement
        for optimal audio quality while respecting sentence boundaries.
        
        Parameters
        ----------
        text : str
            The text to split into groups
        min_chars_per_sentence : int
            Minimum number of characters per sentence group
            
        Returns
        -------
        list
            List of sentence groups optimized for audio generation
        """
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        audio_groups = []
        i = 0
        
        while i < len(sentences):
            current_group = sentences[i]
            j = i + 1
            
            # Continue adding subsequent sentences until min_chars_per_sentence is exceeded
            while j < len(sentences):
                next_sentence = sentences[j]
                combined_length = len(current_group + ". " + next_sentence)
                
                # If the combination exceeds the limit, stop
                if combined_length > min_chars_per_sentence:
                    break
                
                # Otherwise, join the next sentence
                current_group += ". " + next_sentence
                j += 1
            
            audio_groups.append(current_group)
            # Move to the next unprocessed sentence
            i = j
        
        return audio_groups
        
    def _generate(self, text):
        """
        Generate audio from text using the configured provider.
        
        Parameters
        ----------
        text : str
            The text to convert to speech
            
        Returns
        -------
        bytes
            Generated audio data as bytes
        """
        if self._provider == "elevenlabs":
            response = self._client.text_to_speech.convert(
                text=text,
                voice_id=self._voice,
                model_id=self._model,
                output_format="mp3_44100_128",
            )
            response_bytes = b""
            for r in response:
                response_bytes += r
            return response_bytes
        else:
            response = speech(
                model=self._provider+"/"+self._model,
                voice=self._voice,
                input=text,
            )
            return response.content
    
    async def stream(self, text: str, min_chars_per_sentence=100, return_type="audio.mp3"):
        """
        Stream audio generation for long texts.
        
        Generates audio in chunks for long texts, yielding each chunk as it's
        generated. This is useful for real-time audio generation or processing
        very long texts.
        
        Parameters
        ----------
        text : str
            The text to convert to speech
        min_chars_per_sentence : int, optional
            Minimum number of characters per sentence group for optimal audio
            generation (default: 100)
        return_type : str, optional
            The format to return each audio chunk in. Options:
            - File path ending with '.mp3' or '.wav' to save each chunk to file
            - 'base64' to return each chunk as base64 encoded string
            - 'bytes' to return each chunk as raw bytes (default: "audio.mp3")
            
        Yields
        ------
        str or bytes
            Audio chunks in the specified format:
            - If return_type is a file path: yields the file path for each chunk
            - If return_type is 'base64': yields base64 encoded strings
            - If return_type is 'bytes': yields raw audio bytes
            
        Raises
        ------
        ValueError
            If return_type is not a valid option
            
        Examples
        --------
        >>> model = VoiceModel(provider="elevenlabs", model="eleven_multilingual_v2", voice="21m00Tcm4TlvDq8ikWAM")
        >>> async for audio_chunk in model.stream("Long text to convert to speech"):
        ...     print(f"Generated audio chunk: {audio_chunk}")
        """
        audio_groups = self._generate_audio_groups(text, min_chars_per_sentence)
        
        for _, group in enumerate(audio_groups):
            # Generate audio for the sentence group
            response = self._generate(group)

            if return_type.endswith(".mp3") or return_type.endswith(".wav"):
                with open(return_type, 'wb') as f:
                    f.write(response)
                yield return_type
            elif return_type == "base64":
                yield base64.b64encode(response).decode('utf-8')
            elif return_type == "bytes":
                yield response
            else:
                raise ValueError(f"Invalid return type: {return_type}")
