import requests

url = "http://localhost:8000/analyze-project"
# file_path = "Restaurantsolution.zip"
file_path = "Restaurantsolution2.zip"
# file_path = "destinationsolution.zip"
# file_path = "contactbook.zip"

with open(file_path, "rb") as f:
    files = {"zip_file": (file_path, f)}
    response = requests.post(url, files=files)

print(response.json())