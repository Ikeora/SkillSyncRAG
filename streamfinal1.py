import streamlit as st
import matplotlib.pyplot as plt
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
from typing import List
import pandas as pd
import json 
import os
import re
from dotenv import load_dotenv
load_dotenv()

DATABASE=r"C:\Users\ekene\OneDrive\Documents\DataScienceProjects\skill-importance-project\prototype\db\chroma_db_with_metadata"
persistent_directory=DATABASE

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

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


def should_plot(query: str) -> bool:
    plot_keywords = ["plot", "graph", "visualize", "chart", "show","display"]
    return any(keyword in query.lower() for keyword in plot_keywords)

# Function to retrieve job postings (e.g., ChromaDB vector store)
def retrieve_job_postings(query, db):

    # Retrieve relevant documents based on the query
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 20, "score_threshold": 0.25}
    )
    relevant_docs = retriever.invoke(query)
    
    # Inspect the structure of the returned documents
    if relevant_docs:
        # Print the first document to understand its structure
        st.write("Document structure:", relevant_docs[0])
    
    # Adjust this line based on the actual attribute that contains the text
    return '\n'.join([doc.page_content for doc in relevant_docs])  # Replace 'text' with the correct attribute



# Function to extract skills from the job posting dynamically (LLM-based)
def extract_skills_from_job_posting(job_posting: str) -> SkillsResponse:
    # Define the prompt template
    template = """
    Extract the top 10 skills from all jobs provided and their importance (on a scale from 1 to 100):

    {job_posting}

    Return the skills in this format: [{{'skill': 'Skill Name', 'importance': 95}}, ...]
    """
    # Create a PromptTemplate instance
    prompt_template = PromptTemplate(
        input_variables=["job_posting"],  # Define the placeholder(s)
        template=template
    )
    prompt = prompt_template.format(job_posting=job_posting)


    # Use the LLM (ChatGPT) to extract the skills and importance scores
    response = ChatOpenAI(model="gpt-4o").invoke(prompt)

    # Parse the response into structured data (SkillsResponse format)
    pattern = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
    match = pattern.search(response.content)
    if match:
        json_str = match.group(1) # This is the JSON string
        try:
            json_str=json_str.replace("'", '"')
            skills_data = json.loads(json_str) # Convert JSON string to Python object (list of dicts)
        except json.JSONDecodeError as e:
            print("JSON decode error:", e)
    else:
        print("No JSON content was found.")
    
    # Validate and structure the data using Pydantic
    structured_skills = SkillsResponse.model_validate({"skills": 
                                                       [SkillData(skill=skill['skill'], 
                                                                  importance=skill['importance']) 
                                                                  for skill in skills_data]})
    
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

""" query_tool = Tool(
    name="Job Query Responder",
    func=retrieve_job_postings,
    description="Responds with relevant job postings based on the query."
) """

# Set up LangChain agent
llm = ChatOpenAI(model="gpt-4o")
tools = [extract_skills_tool, plot_tool]

# Initialize agent with tools
agent = initialize_agent(tools, llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Streamlit input for query
query = st.text_input("Ask a question:")


if query:
    # Retrieve job postings
    job_postings = retrieve_job_postings(query, db)
    
    # Extract skills if needed
    if should_plot(query):
        skills_response = extract_skills_from_job_posting(job_postings)
        plot_skills(skills_response)
    else:
        # Use the LLM to provide a textual response
        response = llm.invoke(query)
        st.write(response.content)

