import streamlit as st
import matplotlib.pyplot as plt
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
from typing import List
import pandas as pd
import json
import re
from dotenv import load_dotenv
import ast

load_dotenv()

# Set up ChromaDB with embeddings
DATABASE = r"C:\Users\ekene\OneDrive\Documents\DataScienceProjects\skill-importance-project\prototype\db\chroma_db_with_metadata"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

db = Chroma(persist_directory=DATABASE, embedding_function=embeddings, collection_name="jobs_collection")

# Data structure for skills response
class SkillData(BaseModel):
    skill: str
    importance: int

class SkillsResponse(BaseModel):
    skills: List[SkillData]

# Function to retrieve job postings
def retrieve_job_postings(query: str) -> str:
    retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 10, "score_threshold": 0.25})
    relevant_docs = retriever.invoke(query)
    
    if not relevant_docs:
        return "No job postings found."

    return "\n".join([doc.page_content for doc in relevant_docs])

# Function to extract skills using LLM
def extract_skills_from_job_posting(job_posting: str) -> SkillsResponse:
    template = """
    Extract the top 10 skills from the provided job postings and rank them based on their importance (1-100).
    
    Job postings:
    {job_posting}
    
    Provide the result as a valid JSON list with this format:
    ```json
    [{{"skill": "Skill Name", "importance": 95}}, ...]
    ```
    """
    prompt_template = PromptTemplate(input_variables=["job_posting"], template=template)
    prompt = prompt_template.format(job_posting=job_posting)

    response = ChatOpenAI(model="gpt-4o").invoke(prompt)

    # Extract JSON from response
    pattern = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
    match = pattern.search(response.content)
    if match:
        json_str = match.group(1)
        try:
            skills_data = json.loads(json_str)
        except json.JSONDecodeError:
            return SkillsResponse(skills=[]).model_dump_json()
    else:
        return SkillsResponse(skills=[]).model_dump_json()

    skills_response = SkillsResponse(
        skills=[
            SkillData(
                skill=item["skill"], 
                importance=item["importance"]) 
                for item in skills_data])
    
    return skills_response.model_dump_json()

# Function to plot skills

def plot_skills(skills_response_json: str):
    try:
        # Parse the JSON string into a dictionary
        skills_dict = json.loads(skills_response_json)
        # Convert the dictionary to a SkillsResponse object
        skills_response = SkillsResponse(**skills_dict)
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError("Invalid JSON format for skills_response") from e

    # Extract the list of SkillData objects from the SkillsResponse
    skills = skills_response.skills

    # Convert skills to a DataFrame
    df = pd.DataFrame({
        "Skill": [s.skill for s in skills],
        "Importance": [s.importance for s in skills]
    })

    # Create the horizontal bar chart
    plt.figure(figsize=(8, 6))
    plt.barh(df["Skill"], df["Importance"], color="skyblue")
    plt.xlabel("Importance")
    plt.title("Top Skills for Job Role")
    plt.gca().invert_yaxis()  # Show the highest-ranked skill on top
    st.pyplot(plt)
    plt.close()  # Prevent multiple overlapping plots in Streamlit


# Define tools
extract_skills_tool = Tool(
    name="Skill Extractor", 
    func=extract_skills_from_job_posting, 
    description="Extracts skills from job postings.")

plot_tool = Tool(
    name="Top Skills Plotter", 
    func=plot_skills, 
    description="Generates a bar chart of the extracted skills.")

job_retrieval_tool = Tool(
    name="Job Postings Retriever", 
    func=retrieve_job_postings, 
    description="Retrieves job postings based on a query.")

# Agent system message
system_message = """
You are an AI assistant that retrieves job postings, extracts skills, and visualizes them. 
Follow these steps:
1. Retrieve job postings.
2. Extract skills and rank them.
3. If requested, plot the skills.

Use the available tools appropriately.
"""

# Initialize agent
llm = ChatOpenAI(model="gpt-4o")
tools = [job_retrieval_tool, extract_skills_tool, plot_tool]

agent = initialize_agent(
    tools, 
    llm, 
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True, 
    system_message=system_message,
    handle_parsing_errors=True)

# Streamlit UI
query = st.text_input("Enter a job-related query (e.g., 'Top skills for data scientist'):")

if query:
    result = agent.run(query)
    st.write(result)
