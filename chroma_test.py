import os
import shutil
from chromadb import PersistentClient
from chromadb.utils import embedding_functions

# Absolute path to avoid confusion
CHROMA_PATH = os.path.abspath("./chroma_db")

def test_chroma():
    # 1. Clean up any existing database (forcefully)
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)  # This removes directory and all contents
    
    # 2. Create fresh client
    client = PersistentClient(path=CHROMA_PATH)
    
    # 3. Create and populate a collection
    collection = client.get_or_create_collection(
        name="test",
        embedding_function=embedding_functions.DefaultEmbeddingFunction()
    )
    
    collection.add(
        documents=["This is document 1", "This is document 2"],
        metadatas=[{"source": "test1"}, {"source": "test2"}],
        ids=["id1", "id2"]
    )
    
    # 4. Verify contents
    print("\nCollections:")
    print(client.list_collections())
    
    print("\nCollection contents:")
    print(collection.peek())
    
    # 5. Verify files were created
    print("\nFiles created in chroma_db:")
    if os.path.exists(CHROMA_PATH):
        for root, dirs, files in os.walk(CHROMA_PATH):
            for file in files:
                print(f" - {os.path.join(root, file)}")
    else:
        print("ERROR: No chroma_db directory was created!")

if __name__ == "__main__":
    test_chroma()