"""Test what payload contains"""
import sys
sys.path.insert(0, r'c:\Users\IT\Documents\GitHub Repository\Patient_Feedback')

# Test what act_on_case receives
body = {
    'action': 'REJECT',
    'rejection_text': 'My rejection text'
}

rejection_text = body.get("rejection_text", "")
print(f"Payload: {body}")
print(f"rejection_text from get: '{rejection_text}'")
print(f"Length: {len(rejection_text)}")
