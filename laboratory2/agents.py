from camel.messages import BaseMessage
from camel.agents import ChatAgent

class SimpleAssistantAgent(ChatAgent):
    """
    A simple conversation agent that uses a default model.
    """
    agent_role = BaseMessage.make_assistant_message(
        role_name="simple_conversation_agent",
        content="You are a simple assistant agent.",
    )

    def __init__(self, model, message_window_size: int = 20):
        super().__init__(model=model, 
                         system_message=self.agent_role, message_window_size=message_window_size)
