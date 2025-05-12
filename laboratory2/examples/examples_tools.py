from colorama import Fore
from pydantic import BaseModel
from typing import List

from humanlayer.core.approval import HumanLayer
from camel.toolkits import SearchToolkit, FunctionTool, CodeExecutionToolkit, BrowserToolkit
from camel.agents import ChatAgent
from camel.messages import BaseMessage

from config.config import HUMAN_LAYER_KEY
from model import DefaultModel
from agents import SimpleAssistantAgent

def example_use_search_toolkit():
    """
    Example presenting how to use search toolkit.
    """
    message = "What is a current weather in Poland?"

    simple_agent = SimpleAssistantAgent(DefaultModel.create_openai_model())
    response = simple_agent.step(message)
    
    print(Fore.YELLOW + "Response without search toolkit:")
    print(Fore.GREEN + response.msgs[0].content)

    simple_agent_with_tools = ChatAgent(
        "You are a helpful assistant that communicates weather",
        model=DefaultModel.create_openai_model(),
        tools=[
            FunctionTool(SearchToolkit(timeout=5000).search_google),
        ],
    )    
    response = simple_agent_with_tools.step(message)

    print(Fore.YELLOW + "Response with search toolkit:")
    print(Fore.GREEN + response.msgs[0].content)


def example_execute_code_toolkit():
    """
    Example presenting how to use code toolkit.
    """
    code_prompt = BaseMessage.make_user_message(
        role_name="user",
        content="Write a function that collects the values of the American stock market \
            and calculate the change in the last month. Execute the code for AAPL.",
    )

    simple_agent = SimpleAssistantAgent(DefaultModel.create_openai_model())
    response = simple_agent.step(code_prompt)
    
    print(Fore.YELLOW + "Response without code execution toolkit:")
    for msg in response.msgs:
        print(Fore.GREEN + f"Agent response:\n{msg.content}\n")

    
    simple_agent_with_tools = ChatAgent(
        "You are a helpful programmer",
        model=DefaultModel.create_openai_model(),
        tools=CodeExecutionToolkit(sandbox="subprocess", verbose=True).get_tools()
    )    
    response = simple_agent_with_tools.step(code_prompt)

    print(Fore.YELLOW + "Response with code execution toolkit:")
    for msg in response.msgs:
        print(Fore.GREEN + f"Agent response:\n{msg.content}\n")


def example_use_github_toolkit():
    """
    Example presenting how to use github toolkit.
    """
    browser_prompt = "Go to the website: https://www.ibspan.waw.pl/~paprzyck/mp/cvr/research/agent_platforms_site/agent_platforms.html and fetch names of all platforms that can be used for simulating traffic."

    model = DefaultModel.create_openai_model()

    class PointResponse(BaseModel):
        points: List[str]
        summary: str

    browser_agent = ChatAgent(
        "You are an agent that searches the web pages.",
        model=model,
        tools=BrowserToolkit(headless=False, 
                             channel="chrome", 
                             web_agent_model=model,
                             planning_agent_model=model).get_tools()
    )
    response = browser_agent.step(browser_prompt, response_format=PointResponse)
    print(Fore.GREEN + f"Agent response:\n{response.msgs[0].content}\n")


def example_use_human_in_the_loop():
    """
    Example presenting how to use human in the loop with HumanLayer interface.
    """
    interface = HumanLayer(api_key=HUMAN_LAYER_KEY)

    @interface.require_approval()
    def check_dish(message: str) -> str:
        """
        Function that requires human approval.
        """
        return message

    assistant = ChatAgent(
        "You are a helpful assistant that suggests food recipes",
        model=DefaultModel.create_openai_model(),
        tools=[FunctionTool(check_dish)],
    )    
    response = assistant.step("Suggest me a breakfast dish and request its approval.")

    print(Fore.YELLOW + "Response with human in the loop:")
    print(Fore.GREEN + response.msgs[0].content)

    for call in response.info['tool_calls']:
        print(Fore.YELLOW + f"Received calls: {call}")

    