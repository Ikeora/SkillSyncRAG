# API Keys retrieval
from dotenv import load_dotenv
import os
load_dotenv()

# ChromaDB
import os
import chromadb
from langchain_community.vectorstores import Chroma

from langchain.embeddings import HuggingFaceEmbeddings

# OpenAI & Agentic AI 
from langchain_openai import ChatOpenAI

# Data Wrangling & Cleaning
import re
import pandas as pd
import json

import os
from typing import Type

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from langchain.embeddings import HuggingFaceEmbeddings

# Charting libraries
import plotly.express as px


load_dotenv()

"""TOOLS"""

# 1. --- Database accessing tool --- 
class ChromaDBQueryJobsInput(BaseModel):
    query: str = Field(description="Job Search query")

class ChromaDBQueryJobsTool(BaseTool):
    name: str = "job_search" # The AI does not care where (ChromaDB) the datacomes from!!
    description: str = """SEARCH JOBS FIRST. Finds job postings in database."""
    args_schema: Type[BaseModel] = ChromaDBQueryJobsInput

    def _run(self, query: str) -> str:
        # Set the directory
        current_dir = os.getcwd() # NOTE: use this for .ipynb files
        persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

        # Set the model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

        # Set the database 
        collection_name = "jobs_collection"
        db = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embeddings,
            collection_name=collection_name  # Match the name used in push
        )

        # Set the retriever
        retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 10, "score_threshold": 0.25},  # Lower threshold for testing
        )

        # Query the database via .invoke() 
        relevant_docs = retriever.invoke(query)
        
        rel_docs_content = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Return the content of the relevant docs 
        return rel_docs_content

# 2. --- Skill Ranking Tool --- 
class SkillRankingInput(BaseModel):
    job_descriptions: str = Field(description="Job descriptions from job_search")
    user_job_query: str = Field(description="The user's query to find the skills for their specific job title/field")

class SkillRankingTool(BaseTool):
    name: str = "skill_ranking"
    description: str = """ANALYZE JOB POSTINGS TO IDENTIFY SKILLS. 
                            MUST BE USED AFTER chromadb_search."""
    args_schema: Type[BaseModel] = SkillRankingInput

    def _run(self, job_descriptions, user_job_query) -> str: 
        combined_input = (
        "Here are some documents that might help answer the question: "
        + user_job_query
        + "\n\nRelevant Documents:\n"
        + job_descriptions
        + "\n\nPlease extract the top skills for the following job posting and their importance (on a scale from 1 to 100):\nReturn the skills in this format: [{{'skill': 'Skill Name', 'importance': 95}}, ...]"
        )
        jobs_ranked = ChatOpenAI(model="gpt-4o").invoke(combined_input)
        pattern = r"\{\{'skill': '(.*?)', 'importance': (\d{2})\}\}"
        skills_matched = re.findall(pattern, jobs_ranked.content)
        data = {
            'skill': [],
            'rank': [],
        } # Init empty dict to store data
        for skill, rank in skills_matched: # Go through the matches
            data["skill"].append(skill)
            data["rank"].append(rank)
        sorted_data = pd.DataFrame(data).sort_values(by="rank", ascending=False)
        fig = px.bar(
            sorted_data,
            x="rank",
            y="skill",
            orientation='h',
            title = 'Skill Importance Ranking',
            color_continuous_scale="Bluered"
        )
        fig.update_layout(height=600)

        # Create text explanation
        text_output = f"Top {len(sorted_data)} skills for {user_job_query}:\n" + "\n".join(
            [f"{row.Skill}: {row.Importance}/100" for _, row in sorted_data.iterrows()]
        )
        return f"PLOT_DATA||{fig.to_json()}||TEXT_DATA||{text_output}"


# --- Init all the required tools ---

# Create tools using the Pydantic subclass approach
tools = [
    ChromaDBQueryJobsTool(),
    SkillRankingTool(),
]

"""OpenAI Model"""
llm = ChatOpenAI(model="gpt-4o")

# Pull the prompt template from the hub
prompt = hub.pull("hwchase17/openai-tools-agent") # Prompt template that's more prone to utilizing pydantic tools 

# Create the ReAct agent using the create_tool_calling_agent function
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

# Create the agent executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)

"""
user_request = "Give me rankings of data scientists skills needed to land a data science job. Please plot the output into a graph"

# Test the agent with sample queries
response = agent_executor.invoke({"input": user_request})
print(f"Response for '{user_request}':", response)
"""

def query(user_request):
    response = agent_executor.invoke({"input": user_request})

    try:
        if "||TUPLE_START||" in response['output']:
            plot_data, text_data = response['output'].split("||TUPLE_START||")
            return json.loads(plot_data), text_data
    except Exception as e:
        print(f"Tuple parsing failed: {str(e)}")

    return response['output']