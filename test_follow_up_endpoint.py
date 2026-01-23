import requests

response = requests.get('http://localhost:8000/api/follow-up/actions')
print(f"Status Code: {response.status_code}")
print(f"Response: {response.json()}")
