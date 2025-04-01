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
import os
import re
from dotenv import load_dotenv
import plotly.express as px

load_dotenv()

# Set the database location
DATABASE = r"C:\Users\ekene\OneDrive\Documents\DataScienceProjects\skill-importance-project\prototype\db\chroma_db_with_metadata"
persistent_directory = DATABASE

# Initialize embeddings and the vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings,
    collection_name="jobs_collection"
)

# Pydantic models for structuring skills data
class SkillData(BaseModel):
    skill: str
    importance: int

class SkillsResponse(BaseModel):
    skills: List[SkillData]

# ------------------ Data Retrieval Function ------------------

def retrieve_job_postings(query, db):
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 20, "score_threshold": 0.3}
    )
    relevant_docs = retriever.invoke(query)
    
    if relevant_docs:
        st.write("Document structure:", relevant_docs[0])
    
    return '\n'.join([doc.page_content for doc in relevant_docs])

# ------------------ Skill Extraction Function ------------------

def extract_skills_from_job_posting(job_posting: str) -> SkillsResponse:
    template = """
    Extract the top 10 skills from the following job postings and assign an importance score (between 1 and 100) to each skill.
    
    Job postings:
    {job_posting}
    
    Return the result as a JSON list following this format:
    ```json
    [{{"skill": "Skill Name", "importance": 95}}, ...]
    ```
    """
    prompt_template = PromptTemplate(input_variables=["job_posting"], template=template)
    prompt = prompt_template.format(job_posting=job_posting)
    
    response = ChatOpenAI(model="gpt-4o").invoke(prompt)
    
    pattern = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
    match = pattern.search(response.content)
    if match:
        json_str = match.group(1)
        try:
            skills_data = json.loads(json_str)
            return SkillsResponse(skills=[SkillData(**skill) for skill in skills_data])
        except json.JSONDecodeError as e:
            st.error(f"JSON decode error: {e}")
            st.text(f"Response content: {response.content}")
            raise
    else:
        st.error("No JSON content was found in the response.")
        st.text(f"Response content: {response.content}")
        raise ValueError("No valid skills data found in response")

# ------------------ Plotting Function ------------------

def plot_skills(skills_data: dict):
    """Create a horizontal bar plot for the extracted skills and their importance."""
    try:
        # Convert the input to SkillsResponse if it's a dictionary
        if isinstance(skills_data, dict):
            skills_response = SkillsResponse(**skills_data)
        elif isinstance(skills_data, SkillsResponse):
            skills_response = skills_data
        else:
            raise TypeError("Input must be a dictionary or SkillsResponse object")
            
        skills = [s.skill for s in skills_response.skills]
        importance = [s.importance for s in skills_response.skills]
        
        skills_df = pd.DataFrame({
            "Skill": skills,
            "Importance": importance
        })
        
        fig = px.bar(
            skills_df,
            x="Importance",
            y="Skill",
            orientation='h',
            title="Top Skills for Job Role",
            labels={"Importance": "Importance", "Skill": "Skill"},
            color="Importance",
            color_continuous_scale=px.colors.sequential.Viridis
        )

        fig.update_layout(
            xaxis_title="Importance",
            yaxis_title="Skill",
            yaxis=dict(autorange="reversed"),
            template="plotly_white"
        )

        st.plotly_chart(fig)
        return "Successfully plotted the skills"
    except Exception as e:
        st.error(f"Error in plotting: {str(e)}")
        raise

# ------------------ Define Tools for the Agent ------------------

def extract_skills_wrapper(text: str) -> dict:
    """Wrapper function to ensure we return a dictionary"""
    skills_response = extract_skills_from_job_posting(text)
    return skills_response.model_dump()

def plot_skills_wrapper(data: dict) -> str:
    """Wrapper function to handle the plotting"""
    return plot_skills(data)

extract_skills_tool = Tool(
    name="Skill Extractor",
    func=extract_skills_wrapper,
    description="Extracts the top 10 skills and corresponding importance scores from given job postings."
)

plot_tool = Tool(
    name="Top Skills Plotter",
    func=plot_skills_wrapper,
    description="Generates a horizontal bar plot of skills and their importance scores."
)

# ------------------ Agent Setup ------------------

system_instructions = """
You are an expert in job market analysis. When given job posting data:
1. First extract skills using the Skill Extractor tool
2. Then visualize them using the Top Skills Plotter tool
3. Provide a brief interpretation of the results
"""

llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [extract_skills_tool, plot_tool]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    system_instructions=system_instructions
)

# ------------------ Streamlit Interface ------------------

st.title("Job Skills Analyzer")
query = st.text_input("Enter a job title or description to analyze:")

if query:
    with st.spinner("Analyzing job postings..."):
        try:
            job_postings = retrieve_job_postings(query, db)
            
            if not job_postings:
                st.warning("No relevant job postings found for your query.")
            else:
                final_input = {
                    "input": f"Analyze these job postings: {job_postings}",
                    "query": query
                }
                
                response = agent.run(final_input)
                st.success("Analysis complete!")
                if response:
                    st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")