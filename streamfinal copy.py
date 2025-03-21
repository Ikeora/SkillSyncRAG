import streamlit as st
import matplotlib.pyplot as plt
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool, AgentType
from pydantic import BaseModel
from typing import List
import pandas as pd

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embeddings,
    collection_name="jobs_collection"  # Match the name used in push
    )

# Pydantic model to structure the skills and their importance
class SkillData(BaseModel):
    skill: str
    importance: int

class SkillsResponse(BaseModel):
    skills: List[SkillData]



# Function to retrieve job postings (e.g., ChromaDB vector store)
def retrieve_job_postings(query, vector_store):

    # Retrieve relevant documents based on the query
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 10, "score_threshold": 0.25},  # Lower threshold for testing
    )
    # `k`: Top # of document hits 
    # `score_threshold`: Similarity score from text document to query
    relevant_docs = retriever.invoke(query)


# Function to extract skills from the job posting dynamically (LLM-based)
def extract_skills_from_job_posting(job_posting: str) -> SkillsResponse:
    prompt = f"Extract the top skills for the following job posting and their importance (on a scale from 1 to 100):\n\n{job_posting}\n\nReturn the skills in this format: [{{'skill': 'Skill Name', 'importance': 95}}, ...]"
    
    # Use the LLM (ChatGPT) to extract the skills and importance scores
    response = ChatOpenAI(model="gpt-4").run(prompt)
    
    # Parse the response into structured data (SkillsResponse format)
    skills_data = eval(response)  # Convert the string response into a list of dicts
    
    # Validate and structure the data using Pydantic
    structured_skills = SkillsResponse.parse_obj({"skills": [SkillData(skill=skill['skill'], importance=skill['importance']) for skill in skills_data]})
    
    return structured_skills

# Function to plot the skills and importance
def plot_skills(skills_response: SkillsResponse):
    skills = [skill.skill for skill in skills_response.skills]
    importance = [skill.importance for skill in skills_response.skills]
    
    # Create the plot
    skills_df = pd.DataFrame({
        "Skill": skills,
        "Importance": importance
    })
    
    plt.figure(figsize=(8, 6))
    plt.barh(skills_df["Skill"], skills_df["Importance"], color="skyblue")
    plt.xlabel("Importance")
    plt.title("Top Skills for Job Role")
    st.pyplot(plt)

# Define Tools: one for skill extraction, and one for plotting
extract_skills_tool = Tool(
    name="Skill Extractor",
    func=extract_skills_from_job_posting,
    description="Extracts the top skills and their importance from the job posting."
)

plot_tool = Tool(
    name="Top Skills Plotter",
    func=plot_skills,
    description="Generates a plot of the top skills required for the job role."
)

query_tool = Tool(
    name="Job Query Responder",
    func=retrieve_job_postings,
    description="Responds with relevant job postings based on the query."
)

# Set up LangChain agent
llm = ChatOpenAI(model="gpt-4")
tools = [extract_skills_tool, plot_tool, query_tool]

# Initialize agent with tools
agent = initialize_agent(tools, llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Streamlit input for query
query = st.text_input("Ask a question:")

if query:
    # Use the agent to choose which tools to use based on the query
    response = agent.run(query)
    st.write(response)
