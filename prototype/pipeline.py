# API Keys retrieval
from dotenv import load_dotenv
import os
load_dotenv()

# Data Collection
import requests
import pprint

# Storing Data
import uuid 
import requests
import pinecone
from transformers import AutoTokenizer, AutoModel

# ChromaDB
import os
import chromadb
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

# Data Wrangling & Cleaning
import json
import pandas as pd
import torch
import hashlib
import datetime
import re
from dateutil.relativedelta import relativedelta
import re
import jsonlines

# --- ChromaDB ---
# Get current directory and setup persistent storage
# current_dir = os.path.dirname(os.path.abspath(__file__)) # NOTE: use this for .py files
current_dir = os.getcwd() # NOTE: use this for .ipynb files
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# embeddings = OpenAIEmbeddings(
#     model="text-embedding-3-small"
# )

# Init ChromaDB collection once
collection_name = "jobs_collection"
db = Chroma(
    collection_name=collection_name,
    persist_directory=persistent_directory,
    embedding_function=embeddings
)

# Initialize the text splitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

def generate_hash(text):
    """Generate a unique SHA-256 hash if job ID is missing."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def convert_to_datetime(relative_time_str):
    """
        Converts a relative time string like '3 days ago' into a corresponding 
        datetime object.

        Args:
            relative_time_str (str): A string representing a relative date, such as
                                    '1 day ago', '3 days ago', etc.

        Returns:
            datetime: The corresponding datetime object for the specified relative time.
            
        Example:
            >>> convert_to_datetime('3 days ago')
            datetime.datetime(2025, 3, 3, 15, 30, 00)  # The exact date will depend on the current time
            
            Should return datetime in string to be compatible with FireStore datatypes in MM/DD/YYYY.
    """
   
    # First check if the datatype == null
    if not relative_time_str:
        return "03/07/2025"

    # Get the current date and time
    now = datetime.date.today()
    relative_time_str = relative_time_str.strip()

    # Define regex patterns for different time units

    patterns = {
        'minutes': r'(\d+)\s*minutes?\s*ago',
        'hours': r'(\d+)\s*hours?\s*ago',
        'days': r'(\d+)\s*days?\s*ago',
        'weeks': r'(\d+)\s*weeks?\s*ago',
        'months': r'(\d+)\s*months?\s*ago',
        'years': r'(\d+)\s*years?\s*ago'
    }
    
    # Check each pattern and apply the corresponding datetime.timedelta or relativedelta
    datetime_formatted = re.match(r"\b(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/\d{4}\b", relative_time_str)
    if datetime_formatted:
        return relative_time_str
    
    for unit, pattern in patterns.items():
        match = re.match(pattern, relative_time_str)
        if match:
            value = int(match.group(1))
            if unit == 'minutes':
                return (now - relativedelta(minutes=value)).strftime("%m/%d/%Y")
            elif unit == 'hours':
                return (now - relativedelta(hours=value)).strftime("%m/%d/%Y")
            elif unit == 'days':
                return (now - relativedelta(days=value)).strftime("%m/%d/%Y")
            elif unit == 'weeks':
                return (now - relativedelta(weeks=value)).strftime("%m/%d/%Y")
            elif unit == 'months':
                return (now - relativedelta(months=value)).strftime("%m/%d/%Y")
            elif unit == 'years':
                return (now - relativedelta(years=value)).strftime("%m/%d/%Y")
            
    
    # If no pattern matches, return None
    return (now).strftime("%m/%d/%Y")

# -------------- FOR EVERY API RESPONSE: -------------- 
def push_response_to_database(jobs):
    """ Store jobs in ChromaDB """    
    for job in jobs:
        print(f"Processing job: {job['title']} - ID: {job['id']}")

        # Format the metadata
        text = f"{job['title']} at {job['company']} in {job['location']}. {job['employmentType']} position. Salary: {job['salaryRange']}. {job['description']}"
        job_id = job.get("id") or generate_hash(text)

        # NOTE: ** HAVE NOT IMPLEMENTED A CHECKING ALGO **

        print(f"Chunking, embedding, and storing: {job['title']} - ID: {job['id']}")

        # Format 'datePosted'
        job['datePosted'] = str(convert_to_datetime(job['datePosted']))

        # Prepare metadata
        metadata = {
            "id": job_id,
            "title": job['title'],
            "company": job['company'],
            "description": job['description'],
            "location": job['location'],
            "employmentType": job["employmentType"],
            "datePosted": job.get('datePosted'),
            "salaryRange": job.get('salaryRange'),
        }
        metadata = {k: str(v) for k, v in metadata.items()} # Ensure all metadata values are strings
        
        # Split text and prepare metadata for each chunk
        docs = text_splitter.split_text(text)
        metadatas = [metadata] * len(docs)  # Same metadata for all chunks of this job
        
        # Add documents to the existing collection
        db.add_texts(texts=docs, metadatas=metadatas)

        print(f"Job: {job['title']} - ID: {job_id} was properly inserted")
        print("-------------------------------------------------------------")
    db.persist()
    print("Jobs successfully stored in ChromaDB!")

def save_file(response_formatted):
    with jsonlines.open("saved_data/storage.json", mode="a") as writer:
        writer.write(response_formatted)

def call_jobs_api(
        query="data", location="United States", nextPage="", cycles=1):
    """ 
    call_jobs_api() is to call jobs api, with the respective number of cycles. The user may want to continue from where
    their last query terminated, and may do so via `nextPage`. 
    
    NOTE:
    - The values are set to API & our project's defaults; 'Data' jobs in the 'United States'. By default, the function will retrieve 20 jobs in 2 request cycles.
    - We're also assuming English users only, API responses are in English
    """
    
    url = "https://jobs-api14.p.rapidapi.com/v2/list"

    querystring = {
                "query":query,
                "location":location,
                "autoTranslateLocation":"true",
                "remoteOnly":"false",
                "employmentTypes":"fulltime;parttime;intern;contractor"
                }

    headers = {
        "x-rapidapi-key": os.getenv("JOBS_API_KEY"),
        "x-rapidapi-host": "jobs-api14.p.rapidapi.com"
    }

    for _ in range(cycles):
        try: 
            # Request & Retrieve
            response = requests.get(url, headers=headers, params=querystring)

            # Parse response and place into database 
            response_formatted = response.json()
            push_response_to_database(response_formatted['jobs'])

            # Go to the next page for the next cycle
            nextPage = response_formatted['nextPage']
            querystring['nextPage'] = nextPage

            # Just incase storing into the database fails, we need to save the data: 
            save_file(response_formatted)
        except:
            # Try to atleast save the requested data if this fails
            save_file(response_formatted)
            print("An error occured, the requested data was saved")

    return querystring 

"""
---------------------------------------------------------------------
Invoking the function via main.py
---------------------------------------------------------------------

call = call_jobs_api(query="data engineer", cycles=2)

with jsonlines.open("saved_data/call.txt", mode="a") as writer1:
    writer1.write(call)
    writer1.close() 
"""