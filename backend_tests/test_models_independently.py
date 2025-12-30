"""
Independent Testing Script for STT, Classification, and NER Models
Run this script to test each model endpoint independently before integrating into the Insert page.
"""

import requests
import json
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"

# ANSI color codes for better output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")


def print_success(text):
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_error(text):
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_info(text):
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")


def test_classification(text=None):
    """Test Classification Model"""
    print_header("TESTING CLASSIFICATION MODEL")
    
    # Default sample text if none provided
    if text is None:
        text = "المريض يشكو من ألم شديد في البطن وتأخر في تقديم العلاج"
    
    print_info(f"Testing text: {text}")
    
    try:
        # Test the test endpoint first
        print("\n1. Testing classification service availability...")
        response = requests.get(f"{BASE_URL}/api/classification/test")
        
        if response.status_code == 200:
            print_success("Classification service is operational")
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        else:
            print_error(f"Service test failed with status {response.status_code}")
            return
        
        # Test actual classification
        print("\n2. Running classification...")
        response = requests.post(
            f"{BASE_URL}/api/classification/classify",
            json={"text": text, "explain": True}
        )
        
        if response.status_code == 200:
            print_success("Classification successful!")
            result = response.json()
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print_error(f"Classification failed with status {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print_error("Could not connect to the server. Is it running?")
        print_info("Run: uvicorn main:app --reload")
    except Exception as e:
        print_error(f"Error: {str(e)}")


def test_ner(text=None):
    """Test NER Model"""
    print_header("TESTING NER MODEL")
    
    # Default sample text if none provided
    if text is None:
        text = "المريض أحمد محمد يشكو من ألم في البطن وتم فحصه بواسطة الدكتور خالد في قسم الطوارئ"
    
    print_info(f"Testing text: {text}")
    
    try:
        # Test the test endpoint first
        print("\n1. Testing NER service availability...")
        response = requests.get(f"{BASE_URL}/api/ner/test")
        
        if response.status_code == 200:
            print_success("NER service is operational")
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        else:
            print_error(f"Service test failed with status {response.status_code}")
            return
        
        # Test actual NER extraction
        print("\n2. Running NER extraction...")
        response = requests.post(
            f"{BASE_URL}/api/ner/extract",
            json={"text": text}
        )
        
        if response.status_code == 200:
            print_success("NER extraction successful!")
            result = response.json()
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print_error(f"NER extraction failed with status {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print_error("Could not connect to the server. Is it running?")
        print_info("Run: uvicorn main:app --reload")
    except Exception as e:
        print_error(f"Error: {str(e)}")


def test_stt(audio_file_path=None):
    """Test STT Model"""
    print_header("TESTING STT MODEL")
    
    try:
        # Test the test endpoint first
        print("1. Testing STT service availability...")
        response = requests.get(f"{BASE_URL}/api/stt/test")
        
        if response.status_code == 200:
            print_success("STT service is operational")
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        else:
            print_error(f"Service test failed with status {response.status_code}")
            return
        
        # Test actual transcription if audio file provided
        if audio_file_path is None:
            print_info("\nNo audio file provided. Skipping transcription test.")
            print_info("To test transcription, provide an audio file path:")
            print_info('  test_stt(audio_file_path="path/to/your/audio.wav")')
            return
        
        audio_path = Path(audio_file_path)
        if not audio_path.exists():
            print_error(f"Audio file not found: {audio_file_path}")
            return
        
        print(f"\n2. Running transcription on: {audio_path.name}")
        
        with open(audio_path, 'rb') as audio_file:
            files = {'audio_file': (audio_path.name, audio_file, 'audio/wav')}
            response = requests.post(
                f"{BASE_URL}/api/stt/transcribe",
                files=files
            )
        
        if response.status_code == 200:
            print_success("Transcription successful!")
            result = response.json()
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print_error(f"Transcription failed with status {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print_error("Could not connect to the server. Is it running?")
        print_info("Run: uvicorn main:app --reload")
    except Exception as e:
        print_error(f"Error: {str(e)}")


def test_all():
    """Run all tests with default sample data"""
    test_classification()
    test_ner()
    test_stt()


def interactive_menu():
    """Interactive menu for testing"""
    while True:
        print_header("MODEL TESTING MENU")
        print("1. Test Classification Model")
        print("2. Test NER Model")
        print("3. Test STT Model")
        print("4. Test All Models")
        print("5. Custom Classification Test")
        print("6. Custom NER Test")
        print("7. Custom STT Test")
        print("8. Exit")
        
        choice = input(f"\n{Colors.BOLD}Enter your choice (1-8): {Colors.ENDC}")
        
        if choice == "1":
            test_classification()
        elif choice == "2":
            test_ner()
        elif choice == "3":
            test_stt()
        elif choice == "4":
            test_all()
        elif choice == "5":
            custom_text = input("Enter Arabic text for classification: ")
            test_classification(custom_text)
        elif choice == "6":
            custom_text = input("Enter Arabic text for NER: ")
            test_ner(custom_text)
        elif choice == "7":
            audio_path = input("Enter path to audio file: ")
            test_stt(audio_path)
        elif choice == "8":
            print_info("Exiting...")
            break
        else:
            print_error("Invalid choice. Please try again.")
        
        input(f"\n{Colors.BOLD}Press Enter to continue...{Colors.ENDC}")


if __name__ == "__main__":
    import sys
    
    # Check if server is accessible
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print_success(f"Server is running at {BASE_URL}")
        else:
            print_warning("Server responded with unexpected status")
    except:
        print_error(f"Cannot connect to server at {BASE_URL}")
        print_info("Make sure to start the server with: uvicorn main:app --reload")
        sys.exit(1)
    
    # If arguments provided, run specific tests
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        
        if test_type == "classification":
            text = sys.argv[2] if len(sys.argv) > 2 else None
            test_classification(text)
        elif test_type == "ner":
            text = sys.argv[2] if len(sys.argv) > 2 else None
            test_ner(text)
        elif test_type == "stt":
            audio_path = sys.argv[2] if len(sys.argv) > 2 else None
            test_stt(audio_path)
        elif test_type == "all":
            test_all()
        else:
            print_error(f"Unknown test type: {test_type}")
            print_info("Usage: python test_models_independently.py [classification|ner|stt|all] [data]")
    else:
        # Run interactive menu
        interactive_menu()
