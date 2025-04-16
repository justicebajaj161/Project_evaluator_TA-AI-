import os
import zipfile
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configuration
CHROMA_DB_PATH = os.path.abspath("./chroma_db")
EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize clients
def get_chroma_client():
    return PersistentClient(path=CHROMA_DB_PATH)

def get_openai_client():
    return OpenAI(api_key=OPENAI_API_KEY)

def setup_chromadb(project_path: str) -> int:
    """Process project files and store in ChromaDB"""
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name="code_analysis",
        embedding_function=embedding_functions.DefaultEmbeddingFunction()
    )
    
    code_files = []
    supported_extensions = ('.html', '.js', '.jsx', '.ts', '.tsx', '.css')
    
    # Walk through all directories and subdirectories
    for root, _, files in os.walk(project_path):
        for file in files:
            if file.lower().endswith(supported_extensions):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Store relative path to project root
                        rel_path = os.path.relpath(file_path, project_path)
                        code_files.append({
                            'path': rel_path,
                            'content': content
                        })
                except Exception as e:
                    logger.error(f"Error reading {file_path}: {e}")
                    continue
    
    if not code_files:
        return 0
    
    # Prepare data for ChromaDB
    documents = [f"{f['path']}\n{f['content']}" for f in code_files]
    metadatas = [{"path": f["path"]} for f in code_files]
    ids = [f"id_{i}" for i in range(len(code_files))]
    
    try:
        # Clear existing data
        collection.delete(ids=ids)
        
        # Add new documents
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Added {len(code_files)} documents to ChromaDB")
        return len(code_files)
    except Exception as e:
        logger.error(f"Error adding to ChromaDB: {e}")
        return 0

def analyze_with_ai(project_path: str) -> dict:
    """Analyze the project using OpenAI"""
    client = get_openai_client()
    
    # Get relevant code context from ChromaDB
    collection = get_chroma_client().get_collection("code_analysis")
    results = collection.query(
        query_texts=["What is this web application about and how does it look based on the CSS?"],
        n_results=min(10, collection.count())  # Get up to 10 most relevant documents
    )
    
    context = "\n\n".join(results['documents'][0])
    
    # Generate analysis with OpenAI
    prompt = f"""
    Analyze this web project and provide a detailed summary that includes:
    1. What the web application is about (purpose, main functionality)
    2. The visual style based on CSS (colors, layout, typography)
    3. Key components and their relationships
    4. Give RATING as well out of 100. 
    5. This web project should be an UI for a Restaurant based on HTML, CSS, and JS, call it out if its otherwise.
    6. Look for errors in the code in case the code has error because of the which the web app will not compile straightaway bring the RATING below 20
    7. If the code is not a UI for a restaurant, bring the RATING below 20/100.
    Project code context:
    {context}
    
    Respond with a well-structured markdown formatted response.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    return response.choices[0].message.content

def cleanup_chromadb():
    """Clean up ChromaDB after analysis"""
    try:
        client = get_chroma_client()
        client.delete_collection("code_analysis")
        logger.info("Cleaned up ChromaDB collection")
    except Exception as e:
        logger.error(f"Error cleaning up ChromaDB: {e}")

@app.post("/analyze-project")
async def analyze_project(zip_file: UploadFile = File(...)):
    try:
        # Create temp directory
        temp_dir = "temp_project"
        os.makedirs(temp_dir, exist_ok=True)

        # Save uploaded zip
        zip_path = os.path.join(temp_dir, zip_file.filename)
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(zip_file.file, buffer)

        # Unzip the project
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Find project directory (handle nested zip structures)
        project_dir = temp_dir
        contents = os.listdir(temp_dir)
        
        # If there's only one directory in the zip, use that as project root
        if len(contents) == 1 and os.path.isdir(os.path.join(temp_dir, contents[0])):
            project_dir = os.path.join(temp_dir, contents[0])

        # Process files into ChromaDB
        file_count = setup_chromadb(project_dir)
        if file_count == 0:
            raise HTTPException(status_code=400, detail="No relevant code files found or the file is not compilable")

        # Get AI analysis
        analysis = analyze_with_ai(project_dir)

        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)
        cleanup_chromadb()

        return JSONResponse({
            "status": "success",
            "message": f"Processed {file_count} files",
            "analysis": analysis
        })

    except Exception as e:
        logger.error(f"Error in analyze_project: {str(e)}")
        # Clean up even if there's an error
        shutil.rmtree("temp_project", ignore_errors=True)
        cleanup_chromadb()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)