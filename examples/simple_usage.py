#!/usr/bin/env python3
"""
Simple usage example for the Meeting Assistant
"""

import sys
import os
import json

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from meeting_assistant import MeetingAssistantAgent, create_sample_user_data
from meeting_assistant.config import get_config


def main():
    """Simple example of using the meeting assistant"""
    
    # Load sample data
    df_users = create_sample_user_data()
    print(f"Loaded {len(df_users)} users")
    
    # Get configuration
    config = get_config()
    
    # Initialize the agent
    print("Initializing agent...")
    agent = MeetingAssistantAgent(df_users)  # No need to pass azure_config anymore
    
    # Example requests
    requests = [
        "Arda Orçun ile yarın saat 14:00'de 30 dakikalık proje toplantısı organize et",
        "Şahin ve Ege ile pazartesi sabah 10:00'da 60 dakikalık Q3 bütçe görüşmesi yap",
        "Ali ile toplantı yap"
    ]
    
    # Process each request
    for i, request in enumerate(requests, 1):
        print(f"\n{'='*60}")
        print(f"Request {i}: {request}")
        print(f"{'='*60}")
        
        try:
            # This is the main usage: agent.run()
            result = agent.run(request)
            
            if result.get("subject"):
                print("Meeting created successfully!")
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print(f"Response: {result.get('response', 'No response')}")
                
        except Exception as e:
            print(f"Error: {str(e)}")
    
    print(f"\n{'='*60}")
    print("Example completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main() 