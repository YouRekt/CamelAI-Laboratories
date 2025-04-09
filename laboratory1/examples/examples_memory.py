from camel.messages import BaseMessage
from camel.memories import ChatHistoryMemory

from agents import SimpleConversationAgent
from model import DefaultModel

def example_add_record_to_memory():
    """
    Example showing how agent's memory can guide its decisions.
    It presents how to add new conversation record to the memory.    
    """
    model = DefaultModel.create_openai_model()
    user_message = "Hi! Tell me what is the weather today based on what was discussed."

    # Agent without memory
    agent_no_memory = SimpleConversationAgent(model)
    response = agent_no_memory.step(user_message)
    print(f'[No memory agent]: {response.msgs[0].content}')

    # Agent with memory
    agent = SimpleConversationAgent(model)
    weather_message = BaseMessage.make_user_message(
        role_name="user",
        content="Today the weather is sunny and warm today.",
    )
    agent.update_memory(weather_message, role="user")

    response = agent.step(user_message)
    print(f'[Agent with memory] {response.msgs[0].content}')


def example_reading_and_writing_memory():
    """
    Example presenting how to read and write memory.
    """
    model = DefaultModel.create_custom_openai_model('gpt-4o-mini', n=1)
    user_message = "Hi! Right now you are being used to test Camel AI memory during the agent system laboratories at WUT. \
        Please remember this information."

    agent = SimpleConversationAgent(model)
    response = agent.step(user_message)
    print(f'[Agent] {response.msgs[0].content}')

    agent.save_memory('./memory_dump.json')

    agent = SimpleConversationAgent(model)
    agent.load_memory_from_path('./memory_dump.json')

    user_message_2 = 'Hi! What were we talking about?'
    response = agent.step(user_message_2)
    print(f'[Agent] {response.msgs[0].content}')