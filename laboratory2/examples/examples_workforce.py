from camel.societies.workforce import Workforce
from camel.agents import ChatAgent
from camel.tasks import Task

from camel.toolkits import SearchToolkit, FunctionTool, SemanticScholarToolkit, BrowserToolkit

from model import DefaultModel

async def example_workforce_for_preparing_agentic_computing_website():
    """
    Example demonstrates the usage of agent workflow.
    """
    model = DefaultModel.create_openai_model()

    research_agent = ChatAgent(
        "You are a helpful assistant that searches the research papers",
        model=model,
        tools=SemanticScholarToolkit(timeout=5000).get_tools(),
    )
    web_agent = ChatAgent(
        "You are a helpful assistant that searches the web",
        model=model,
        tools=[FunctionTool(SearchToolkit(timeout=5000).search_google),
        *BrowserToolkit(headless=True, channel="chrome", web_agent_model=model, planning_agent_model=model).get_tools()],
    ) 

    workforce = Workforce(
        "Preparing agentic computing platforms website."
    )
    workforce.add_single_agent_worker("Agent searching web for details.", worker=web_agent)
    workforce.add_single_agent_worker("Agent searching through research papers.", worker=research_agent)

    task = Task(
        content="Prepare a website listing frameworks for agentic computing." \
        "Find frameworks based on the research papers. On the website, include the links to the pages/code repositories of the frameworks." \
        "Moreover, include the information about the latest release of the platform.",
        id="0"
    )

    task = workforce.process_task(task)
    print('Final Result of Original task:\n', task.result)
