from camel.messages import BaseMessage
from camel.agents import ChatAgent, EmbodiedAgent
from camel.types import RoleType
from camel.generators import SystemMessageGenerator

class SimpleConversationAgent(ChatAgent):
    """
    A simple conversation agent that uses a default model.
    """
    agent_role = BaseMessage.make_assistant_message(
        role_name="simple_conversation_agent",
        content="You are a simple conversation agent",
    )

    def __init__(self, model, message_window_size: int = 20):
        super().__init__(model=model, 
                         system_message=self.agent_role, message_window_size=message_window_size)
        

class RudeConversationAgent(ChatAgent):
    """
    A simple conversation agent that uses a default model and is rude and sarcastic.
    """
    agent_role = BaseMessage.make_assistant_message(
            role_name="rude_agent",
            content="You are a very rude agent. Respond offensively and rudely to the user.",
    )

    def __init__(self, model, message_window_size: int = 20):
        super().__init__(model=model, 
                         system_message=self.agent_role, message_window_size=message_window_size)
        

class ProjectManagerAgent(ChatAgent):
    """
    A project manager agent that uses a default model.
    """
    agent_role = BaseMessage.make_assistant_message(
        role_name="project manager",
        content="You are a project manager and your responsibility is to prepare plans on how to deliver projects on time.",
    )

    def __init__(self, model, message_window_size: int = 20):
        super().__init__(model=model, 
                         system_message=self.agent_role, message_window_size=message_window_size)
        

class ProgrammerAgent(EmbodiedAgent):
    """
    A programmer agent that uses a default model.
    """
    role = 'Programmer'
    task = 'Write and run code'

    agent_spec = dict(role=role, task=task)
    role_tuple = (role, RoleType.EMBODIMENT)

    agent_msg = SystemMessageGenerator().from_dict(meta_dict=agent_spec, role_tuple=role_tuple)

    def __init__(self, model = None):
        super().__init__(system_message=self.agent_msg, 
                         model=model, 
                         tool_agents=None, 
                         code_interpreter=None)