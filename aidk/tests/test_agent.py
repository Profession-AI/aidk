from aidk.models import Model
from aidk.agents import Agent
from aidk.agents.agentic_loop import ProgrammaticAgenticLoop
from datetime import datetime
from aidk.tools import domain_whois, search_web


def get_current_time(location: str):

    """
    Get the current time in a given location

    Args:
        location (str): the location
    """

    return datetime.now().strftime("%H:%M")

def get_user_location():
    
    """
    Get the location of the current user
    """
    
    return "Catania"

model = Model(provider="openai", model="gpt-4.1-nano")
agentic_loop = ProgrammaticAgenticLoop(model=model, 
              debug=True,
              max_iter=10)

agent = Agent(model=model, 
              tools=[get_current_time, get_user_location], 
              paradigm=agentic_loop)

resp = agent.run("Che ora è? Fai così: 1 - ottieni la località, 2 - ottieni l'ora")
print(resp)