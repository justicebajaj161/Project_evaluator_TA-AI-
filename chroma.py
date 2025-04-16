from chromadb import Client, Settings
client = Client(Settings(persist_directory="./chroma_db"))
print(client.list_collections())