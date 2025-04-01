import streamlit as st
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
import plotly.express as px

load_dotenv()

# Initialize database and embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
db = Chroma(
    persist_directory=r"C:\Users\ekene\OneDrive\Documents\DataScienceProjects\skill-importance-project\prototype\db\chroma_db_with_metadata",
    embedding_function=embeddings,
    collection_name="jobs_collection"
)

# Pydantic models
class SkillData(BaseModel):
    skill: str
    importance: int

class SkillsResponse(BaseModel):
    skills: List[SkillData]

def retrieve_job_postings(query, db):
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 20, "score_threshold": 0.3}
    )
    relevant_docs = retriever.invoke(query)
    return '\n'.join([doc.page_content for doc in relevant_docs])

def extract_skills(job_posting: str) -> dict:
    template = """Extract top 10 skills and importance scores (1-100) from these job postings:
    {job_posting}
    
    Return JSON format:
    ```json
    {{
        "skills": [
            {{"skill": "Skill 1", "importance": 95}},
            {{"skill": "Skill 2", "importance": 90}}
        ]
    }}
    ```"""
    prompt = PromptTemplate(input_variables=["job_posting"], template=template).format(job_posting=job_posting)
    response = ChatOpenAI(model="gpt-4o").invoke(prompt)
    
    # Extract JSON from response
    match = re.search(r"```json\s*(.*?)\s*```", response.content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            st.error("Failed to parse skills data")
            return {"skills": []}
    return {"skills": []}

def plot_skills(skills_data: dict):
    if not skills_data.get("skills"):
        return "No skills data to plot"
    
    df = pd.DataFrame(skills_data["skills"])
    df = df.sort_values("importance", ascending=False)
    
    fig = px.bar(
        df,
        x="importance",
        y="skill",
        orientation='h',
        title="Top Skills by Importance",
        labels={"importance": "Importance Score", "skill": "Skill"},
        color="importance",
        color_continuous_scale=px.colors.sequential.Viridis
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig)
    return "Skills visualized successfully"

# Create tools
tools = [
    Tool(
        name="SkillExtractor",
        func=extract_skills,
        description="Extracts skills and importance scores from job postings. Returns JSON."
    ),
    Tool(
        name="SkillVisualizer",
        func=plot_skills,
        description="Visualizes skills data. Input should be JSON with 'skills' array."
    )
]

# Initialize agent
agent = initialize_agent(
    tools,
    ChatOpenAI(model="gpt-4o", temperature=0),
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# Streamlit UI
st.title("Job Skills Analyzer")
query = st.text_input("Enter job title or description:")

if st.button("Analyze") and query:
    with st.spinner("Processing..."):
        try:
            postings = retrieve_job_postings(query, db)
            if not postings:
                st.warning("No matching job postings found")
            else:
                # Directly extract and visualize skills
                skills_data = extract_skills(postings)
                if skills_data["skills"]:
                    plot_skills(skills_data)
                    
                    # Display the raw data
                    st.subheader("Extracted Skills Data")
                    st.dataframe(pd.DataFrame(skills_data["skills"]).sort_values("importance", ascending=False))
                    
                    # Get agent's analysis
                    analysis = agent.run({
                        "input": f"Analyze these skills for {query}: {skills_data}",
                        "query": "Provide insights about these skills"
                    })
                    st.write(analysis)
                else:
                    st.warning("No skills could be extracted")
        except Exception as e:
            st.error(f"Error: {str(e)}")