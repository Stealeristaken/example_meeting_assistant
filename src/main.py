#!/usr/bin/env python3
"""
Main script for the Meeting Assistant
"""

import json
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from meeting_assistant import MeetingAssistantAgent, create_sample_user_data, HealthChecker
from meeting_assistant.config import get_config


def main():
    """Main function"""
    print("Enhanced Meeting Assistant")
    print("=" * 50)
    
    # Load sample data
    df_users = create_sample_user_data()
    print(f"Loaded {len(df_users)} users")
    
    # Get configuration
    config = get_config()
    azure_config = config.get_azure_config()
    
    # Run health check
    print("\nRunning health check...")
    health_checker = HealthChecker()
    health_report = health_checker.run_full_health_check()  # No need to pass parameters
    health_checker.print_health_report(health_report)
    
    if health_report['overall_status'] == 'FAIL':
        print("Critical issues detected. Please fix them before proceeding.")
        return
    
    try:
        # Initialize agent
        print("\nInitializing Meeting Assistant Agent...")
        agent = MeetingAssistantAgent(df_users)  # No need to pass azure_config anymore
        
        print("\n" + "="*50)
        print("AGENT READY!")
        print("="*50)
        
        print("\nExample usage:")
        print("• 'Arda Orçun ve Şahin ile yarın 90 dakikalık proje toplantısı organize et'")
        print("• 'Ali ve Ahmet ile pazartesi sabah 10:00'da toplantı planla'")
        print("• 'Ozden ve Emre ile Q3 bütçe görüşmesi yap'")
        print("• 'Yarın saat 14:00'de Arda ile hızlı sync'")
        
        # Interactive mode
        while True:
            print("\n" + "-"*50)
            user_input = input("Enter your request (or 'quit' to exit): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            print("Processing your request...")
            
            # Process request
            result = agent.run(user_input)
            
            # Check if clarification is needed
            if result.get("needs_clarification") or "seçim yapın" in result.get("response", ""):
                print(f"\nAgent: {result.get('response', 'Clarification needed')}")
                
                clarification = input("\nYour choice: ").strip()
                
                if clarification:
                    print("Processing your choice...")
                    final_result = agent.handle_clarification(clarification)
                    
                    if final_result.get("subject"):
                        print("\nMeeting successfully created!")
                        print(json.dumps(final_result, ensure_ascii=False, indent=2))
                    else:
                        print(f"Agent: {final_result.get('response', 'An error occurred')}")
            
            elif result.get("subject"):
                print("\nMeeting successfully created!")
                print(json.dumps(result, ensure_ascii=False, indent=2))
            
            elif result.get("response"):
                print(f"Agent: {result['response']}")
            
            elif result.get("error"):
                print(f"Error: {result['error']}")
            
            else:
                print("Agent: I couldn't process this request. Please try again.")
    
    except Exception as e:
        print(f"Initialization error: {e}")
        print("\nFalling back to vector database test mode...")
        
        # Fallback to vector testing
        from meeting_assistant import VectorDatabaseManager
        
        vector_db = VectorDatabaseManager(df_users)
        
        while True:
            print("\n" + "-"*30)
            user_input = input("Enter names to search (or 'quit'): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            names = [n.strip() for n in user_input.split(',') if n.strip()]
            if not names:
                continue
            
            print(f"\nSearching for: {names}")
            results = vector_db.search_names(names)
            
            print(f"Needs clarification: {results['needs_clarification']}")
            
            for resolved in results['resolved_names']:
                score = resolved.get('similarity_score', 'N/A')
                print(f"Resolved '{resolved['input_name']}' → {resolved['matched_user']['full_name']} (score: {score})")
            
            for partial in results['partial_matches']:
                print(f"Ambiguous '{partial['input_name']}' - {len(partial['candidates'])} candidates:")
                for i, candidate in enumerate(partial['candidates'], 1):
                    score = candidate.get('similarity', 'N/A')
                    print(f"   {i}. {candidate['full_name']} (score: {score})")
    
    print("\nGoodbye!")


if __name__ == "__main__":
    main() 