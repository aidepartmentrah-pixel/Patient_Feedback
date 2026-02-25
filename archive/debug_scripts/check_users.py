import sys
from pathlib import Path

backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from backend.api.db_layer.auth_db import get_all_users

users = get_all_users()
print("\nUsers in database:")
for u in users:
    print(f"  - {u['username']}")
print(f"\nTotal: {len(users)} users\n")
