from model import DefaultModel
from agents import PreferenceAgent, JobSearchAgent, WebAgent, CodingAgent
from camel.societies.workforce import Workforce
from camel.tasks import Task
from camel.messages import BaseMessage

model = DefaultModel.create_openai_model()

preference_agent = PreferenceAgent(model)
job_search_agent = JobSearchAgent(model)
web_agent = WebAgent(model)
coding_agent = CodingAgent(model)

workforce = Workforce("Job search and interview preparation")

workforce.add_single_agent_worker("Job Preference Agent: Collects user preferences for job search including position level, salary requirements, locations and technical stack",worker=preference_agent)
workforce.add_single_agent_worker("Job Search Agent: Searches and filters job positions matching user preferences, sort results by salary in descending order and extract key details like position name, salary, location and application link",worker=job_search_agent)
workforce.add_single_agent_worker("Web Agent: Finds relevant interview preparation resources and creates a detailed 2-week preparation plan with daily goals, timeline and learning materials for both theoretical and practical interviews",worker=web_agent)
# workforce.add_single_agent_worker("Coding Agent: Creates and serves an HTML website using Flask to present job search results and planning information",worker=coding_agent)

task = Task(content="""
    1. Collect detailed user preferences for the job search
    2. Search for matching positions and return results sorted by salary (descending) with:
       - Position name
       - Salary
       - Location  
       - Application link
    3. Find interview preparation resources and create:
       - Summary of key learning materials
       - 2-week preparation plan with:
         * Daily goals and timeline
         * Specific resources to study
         * Balance of theoretical and practical preparation
    """,
    id="0"
)

task = workforce.process_task(task)

print(task.result)

prompt = BaseMessage.make_user_message(
    role_name="workforce",
    content=f"""
    Use this data to populate the website: {task.result}
    """
)

response = coding_agent.step(prompt)

print(response.msg.content)

input("Press Enter to exit...")
