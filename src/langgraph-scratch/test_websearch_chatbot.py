import requests
import json
import time
import pytest
from typing import Dict, Any

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_THREAD_ID = "test_thread_123"

class WebSearchChatbotTester:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
    
    def chat(self, message: str, thread_id: str = TEST_THREAD_ID) -> Dict[str, Any]:
        """Send a chat message to the API."""
        response = self.session.post(
            f"{self.base_url}/chat",
            json={"message": message, "thread_id": thread_id}
        )
        response.raise_for_status()
        return response.json()
    
    def human_response(self, response: str, thread_id: str = TEST_THREAD_ID) -> Dict[str, Any]:
        """Send human response to continue interrupted conversation."""
        response_obj = self.session.post(
            f"{self.base_url}/human_response",
            json={"response": response, "thread_id": thread_id}
        )
        response_obj.raise_for_status()
        return response_obj.json()
    
    def get_thread_state(self, thread_id: str = TEST_THREAD_ID) -> Dict[str, Any]:
        """Get current thread state."""
        response = self.session.get(f"{self.base_url}/threads/{thread_id}/state")
        response.raise_for_status()
        return response.json()
    
    def clear_thread(self, thread_id: str = TEST_THREAD_ID) -> Dict[str, Any]:
        """Clear a thread."""
        response = self.session.delete(f"{self.base_url}/threads/{thread_id}")
        response.raise_for_status()
        return response.json()

def test_basic_chat():
    """Test basic chat functionality."""
    tester = WebSearchChatbotTester()
    
    # Clear thread first
    tester.clear_thread()
    
    # Send a simple message
    result = tester.chat("Hello, how are you?")
    
    print("Basic Chat Test:")
    print(f"Status: {result['status']}")
    print(f"Messages: {len(result['messages'])}")
    print(f"Last message: {result['messages'][-1]['content'][:100]}...")
    
    assert result["status"] in ["completed", "interrupted"]
    assert len(result["messages"]) > 0

def test_web_search():
    """Test web search functionality."""
    tester = WebSearchChatbotTester()
    
    # Clear thread first
    tester.clear_thread()
    
    # Ask a question that should trigger web search
    result = tester.chat("What's the latest news about artificial intelligence?")
    
    print("\nWeb Search Test:")
    print(f"Status: {result['status']}")
    print(f"Messages: {len(result['messages'])}")
    if result['messages']:
        print(f"Last message: {result['messages'][-1]['content'][:200]}...")
    
    assert result["status"] in ["completed", "interrupted"]

def test_human_assistance_workflow():
    """Test human assistance interruption workflow."""
    tester = WebSearchChatbotTester()
    
    # Clear thread first
    tester.clear_thread()
    
    # Send a message that might trigger human assistance
    result = tester.chat("I need help with a complex decision. Please ask a human for advice on whether I should invest in stocks or bonds given current market conditions.")
    
    print("\nHuman Assistance Test:")
    print(f"Status: {result['status']}")
    
    if result["status"] == "interrupted":
        print(f"Interruption query: {result.get('interruption_query', 'N/A')}")
        
        # Provide human response
        human_result = tester.human_response("Based on current market volatility, I'd recommend a balanced portfolio with 60% stocks and 40% bonds.")
        
        print(f"After human response - Status: {human_result['status']}")
        print(f"Final messages: {len(human_result['messages'])}")
        
        assert human_result["status"] in ["completed", "interrupted"]
    else:
        print("No interruption occurred - that's also valid behavior")

def test_thread_state():
    """Test thread state management."""
    tester = WebSearchChatbotTester()
    
    # Clear thread first
    tester.clear_thread()
    
    # Check initial state
    state = tester.get_thread_state()
    print("\nThread State Test:")
    print(f"Initial state - Messages: {state['message_count']}, Interrupted: {state['is_interrupted']}")
    
    # Send a message
    result = tester.chat("Tell me about Python programming")
    
    # Check state after message
    state = tester.get_thread_state()
    print(f"After message - Messages: {state['message_count']}, Interrupted: {state['is_interrupted']}")
    
    assert state['message_count'] > 0

def test_multiple_threads():
    """Test multiple conversation threads."""
    tester = WebSearchChatbotTester()
    
    thread1 = "thread_1"
    thread2 = "thread_2"
    
    # Clear both threads
    tester.clear_thread(thread1)
    tester.clear_thread(thread2)
    
    # Send different messages to different threads
    result1 = tester.chat("My name is Alice", thread1)
    result2 = tester.chat("My name is Bob", thread2)
    
    # Check thread states
    state1 = tester.get_thread_state(thread1)
    state2 = tester.get_thread_state(thread2)
    
    print("\nMultiple Threads Test:")
    print(f"Thread 1 messages: {state1['message_count']}")
    print(f"Thread 2 messages: {state2['message_count']}")
    
    # Send follow-up messages
    result1_followup = tester.chat("What's my name?", thread1)
    result2_followup = tester.chat("What's my name?", thread2)
    
    print("Follow-up responses:")
    if result1_followup['messages']:
        print(f"Thread 1 response: {result1_followup['messages'][-1]['content'][:100]}...")
    if result2_followup['messages']:
        print(f"Thread 2 response: {result2_followup['messages'][-1]['content'][:100]}...")

def test_error_handling():
    """Test error handling."""
    tester = WebSearchChatbotTester()
    
    # Clear thread first
    tester.clear_thread()
    
    # Try to send human response when not interrupted
    try:
        result = tester.human_response("This should fail")
        print("\nError Handling Test:")
        print(f"Human response when not interrupted: {result}")
        assert result["status"] == "error"
    except requests.exceptions.HTTPError as e:
        print(f"Expected HTTP error: {e}")

def run_interactive_test():
    """Run an interactive test session."""
    tester = WebSearchChatbotTester()
    
    print("\n" + "="*50)
    print("INTERACTIVE TEST SESSION")
    print("="*50)
    
    # Clear thread
    tester.clear_thread("interactive_test")
    
    while True:
        try:
            user_input = input("\nEnter message (or 'quit' to exit): ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            result = tester.chat(user_input, "interactive_test")
            
            print(f"\nStatus: {result['status']}")
            
            if result['status'] == 'interrupted':
                print(f"Interruption: {result.get('interruption_query', 'Human assistance needed')}")
                human_input = input("Provide human response: ")
                
                human_result = tester.human_response(human_input, "interactive_test")
                print(f"After human response - Status: {human_result['status']}")
                
                # Print final messages
                for msg in human_result['messages'][-2:]:
                    print(f"{msg['role']}: {msg['content']}")
            else:
                # Print messages
                for msg in result['messages'][-2:]:
                    print(f"{msg['role']}: {msg['content']}")
                    
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    print("Web Search Chatbot API Tester")
    print("Make sure the server is running at http://localhost:8000")
    print("Start server with: python websearch_chatbot.py server")
    
    try:
        # Run basic tests
        test_basic_chat()
        test_web_search()
        test_human_assistance_workflow()
        test_thread_state()
        test_multiple_threads()
        test_error_handling()
        
        print("\n" + "="*50)
        print("ALL TESTS COMPLETED")
        print("="*50)
        
        # Ask if user wants interactive mode
        interactive = input("\nRun interactive test? (y/n): ")
        if interactive.lower() == 'y':
            run_interactive_test()
            
    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not connect to server at http://localhost:8000")
        print("Please start the server first with: python websearch_chatbot.py server")
    except Exception as e:
        print(f"\nTest failed with error: {e}")