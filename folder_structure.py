import os

def create_folders(folder_structure, base_path="."):
    for folder in folder_structure:
        path = os.path.join(base_path, folder)
        
        # Check if it's a directory or a file
        if not os.path.splitext(folder)[1]:  # No file extension means it's a directory
            os.makedirs(path, exist_ok=True)
        else:
            # Ensure the directory exists before creating the file
            dir_name = os.path.dirname(path)
            os.makedirs(dir_name, exist_ok=True)
            try:
                with open(path, "w") as f:
                    f.write("# Placeholder file")
            except PermissionError as e:
                print(f"PermissionError: Unable to write to {path}. {e}")
                continue  # Skip the file if permission error occurs

if __name__ == "__main__":
    folder_structure = [
        "data/raw",
        "data/processed",
        "embeddings",
        "vector_store",
        "src/api",
        "src/api/fetch_jobs.py",
        "src/api/preprocess.py",
        "src/api/store_firestore.py",
        "src/embeddings",
        "src/embeddings/generate.py",
        "src/embeddings/query.py",
        "src/llm",
        "src/llm/generate_response.py",
        "src/llm/prompt_templates.py",
        "src/rag",
        "src/rag/retriever.py",
        "src/rag/generator.py",
        "src/rag/pipeline.py",
        "src/utils",
        "src/utils/config.py",
        "src/utils/logging.py",
        "tests",
        "tests/test_embeddings.py",
        "tests/test_rag.py",
        "tests/test_llm.py",
        "notebooks",
        "configs",
        "configs/chromadb.yaml",
        "configs/firebase.json",
        "configs/model.yaml",
        "requirements.txt",
    ]
    
    create_folders(folder_structure, base_path='.')
    print("Project structure has been created successfully.")
