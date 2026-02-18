"""
Rename get_db_connection() to get_connection() in all api_v2/db_layer files
"""

import os
import glob

# Get all Python files in api_v2/db_layer
files = glob.glob("api_v2/db_layer/**/*.py", recursive=True)

modified_count = 0

for filepath in files:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Count occurrences
        occurrences = content.count('get_db_connection()')
        
        if occurrences > 0:
            # Replace all occurrences
            new_content = content.replace('get_db_connection()', 'get_connection()')
            
            # Write back
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print(f"✓ {filepath}: {occurrences} replacements")
            modified_count += 1
    
    except Exception as e:
        print(f"✗ {filepath}: {e}")

print(f"\n✅ Modified {modified_count} files")
