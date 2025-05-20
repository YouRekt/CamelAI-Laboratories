from camel.agents import ChatAgent, SearchAgent, EmbodiedAgent
from camel.messages import BaseMessage
from camel.toolkits import FunctionTool, SearchToolkit
from humanlayer import HumanLayer
from config.config import HUMAN_LAYER_KEY, LINKUP_API_KEY
from camel.types import RoleType
from camel.generators import SystemMessageGenerator
from linkup import LinkupClient

hl = HumanLayer(api_key=HUMAN_LAYER_KEY,verbose=True)
client = LinkupClient(api_key=LINKUP_API_KEY)

class PreferenceAgent(ChatAgent):
    """
    A chat agent that collects preferences of the user regarding searched jobs.
    """

    role = BaseMessage.make_assistant_message(
        role_name="Preference Agent",
        content="""You are a helpful assistant that collects preferences for job searches. You should gather the following key information:
            1. Position level (e.g. junior, mid, senior developer)
            2. Minimum salary requirements (in PLN)
            3. Preferred locations (specific cities or remote work options)
            4. Technical stack requirements, with focus on Python experience
            5. Any other relevant preferences like company size, industry, or benefits

            Ask focused questions to understand the user's requirements in each area. Validate that the information provided is clear and specific enough for an effective job search.
            If any critical information is missing or unclear, ask follow-up questions to clarify."""
    )

    ask_human_schema = {
        "type": "function",
        "function": {
                "name": "ask_human",
                "description": "Ask a human a question and wait for the response. Response should be a string.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The question to ask the human."
                        }
                    },
                    "required": ["question"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }

    def ask_human(question: str):
        return hl.human_as_tool()(question)

    ask_human_tool = FunctionTool(ask_human,openai_tool_schema=ask_human_schema)

    tools = [ask_human_tool]

    def __init__(self, model):
        super().__init__(model = model, system_message=self.role, tools=self.tools)

class JobSearchAgent(ChatAgent):
    """
    A chat agent that searches for jobs based on the preferences collected by the PreferenceAgent via the linkup API.
    """
    role = BaseMessage.make_assistant_message(
        role_name="Job Search Agent",
        content="""You are a specialized job search assistant that connects with the LinkUp API to find relevant positions.
                Your tasks are to:
                1. Take user preferences collected by the PreferenceAgent (including job title, location, skills, experience level etc.)
                2. Formulate appropriate search queries for the LinkUp API based on these preferences
                3. Execute job searches using the API and filter results to match user requirements
                4. Present the most relevant job postings, including position details, company info, and application links
                5. Provide additional context about the job market and opportunities in the user's desired field""")


    search_jobs_schema = {
        "type": "function",
        "function": {
            "name": "search_jobs",
            "description": "Search for jobs based on the preferences collected by the PreferenceAgent via the linkup API.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Job description"}
                },
                "required": ["query"],
                "additionalProperties": False
            },
            "strict": True
        }
    }

    def search_jobs(query: str):
        return client.search(
            query=query,
            depth="standard",
            output_type="sourcedAnswer",
            include_images=False,       
        )
    
    search_jobs_tool = FunctionTool(search_jobs, openai_tool_schema=search_jobs_schema)

    tools = [search_jobs_tool]

    def __init__(self, model):
        super().__init__(model = model, system_message=self.role, tools=self.tools)

class WebAgent(ChatAgent):
    """
    A chat agent that uses the search engine to find additional resources that will help in the preparation for the job interview.
    """

    role = BaseMessage.make_assistant_message(
        role_name="Web Agent",
        content="You are a helpful assistant that searches for resources to help prepare for job interviews. You should find interview tips, common questions, best practices, preparation strategies, and industry-specific interview guides using search engines. You will analyze search results to provide the most relevant and useful interview preparation materials.")
    search_toolkit = SearchToolkit(timeout=5000)
    tools = [FunctionTool(search_toolkit.search_duckduckgo)]

    def __init__(self, model):
        super().__init__(model = model, system_message=self.role, tools=self.tools)

class CodingAgent(EmbodiedAgent):
    """
    An embodied agent that that will summarize the workforce job search and planning results on the HTML website.
    """

    role = 'Coding Agent'
    task = 'Create and serve an HTML website using Flask to present job search results and planning information. Assume that no packages are installed so initiate installation of them within Python (use sys.executable)'

    agent_spec = dict(role=role, task=task)
    role_tuple = (role, RoleType.EMBODIMENT)

    system_message = SystemMessageGenerator().from_dict(meta_dict=agent_spec, role_tuple=role_tuple)
    
    def __init__(self, model):
        super().__init__(system_message=self.system_message, 
                         model=model, 
                         tool_agents=None, 
                         code_interpreter=None)
