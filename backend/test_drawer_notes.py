"""
Test drawer notes database layer
"""
from api_v2.db_layer.drawer_note_db import list_notes_paged

try:
    notes = list_notes_paged(limit=10, offset=0)
    print(f"SUCCESS: Retrieved {len(notes)} notes")
    for note in notes[:3]:
        text = note["note_text"][:50] if len(note["note_text"]) > 50 else note["note_text"]
        print(f"  - Note {note['note_id']}: {text}...")
except Exception as e:
    print(f"ERROR: {e}")
