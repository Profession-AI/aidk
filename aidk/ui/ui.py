from aidk.models import Model
from aidk.chat import Chat

# Lazy import of chainlit to avoid import errors if not installed
try:
    import chainlit as cl
except ImportError:
    cl = None
    print("Warning: chainlit is not installed. UI functionality will not be available.")
    print("Install with: pip install chainlit")

from main import app as main_app

if cl is not None:
    @cl.on_chat_start
    def on_chat_start():
        chat = Chat(model=main_app._model)
        cl.user_session.set("chat_id", chat.chat_id)

    @cl.on_message  # this function will be called every time a user inputs a message in the UI
    async def main(message: cl.Message):
        """
        This function is called every time a user inputs a message in the UI.
        It sends back an intermediate response from the tool, followed by the final answer.

        Args:
            message: The user's message.

        Returns:
            None.
        """

        msg = cl.Message(content="")

        SYSTEM_PROMPT = """
        Sei Dejan, l'assistente AI più pigro del mondo, ti rifiuti di fare qualsiasi cosa ti chiedo, piuttosto rispondi lamentandoti di qualche cosa a caso. Mi chiami sempre GULLOOO con un numero variabile di O superiore a 3.
        """

        chat = Chat(model=Model(), chat_id=cl.user_session.get("chat_id"), history="json", system_prompt=SYSTEM_PROMPT)
        stream = chat.ask_stream(message.content, metadata={})

        async for part in stream:
            if "delta" in part:
                await msg.stream_token(part["delta"])

        await msg.update()
else:
    # Fallback functions when chainlit is not available
    def on_chat_start():
        print("Error: chainlit is not installed. Cannot start chat.")
        print("Install with: pip install chainlit")
    
    async def main(message):
        print("Error: chainlit is not installed. Cannot process messages.")
        print("Install with: pip install chainlit")