import sys
import json
sys.path.insert(0, 'backend')

from api.db_layer.explanation_seasonal_db import get_seasonal_reports_needing_explanation

# Test the function directly
result = get_seasonal_reports_needing_explanation()

print(json.dumps(result, indent=2, ensure_ascii=False))
