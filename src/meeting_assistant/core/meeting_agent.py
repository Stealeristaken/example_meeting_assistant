"""
Main meeting assistant agent
"""

import json
import re
from typing import Dict, Any, Optional
from langchain_openai import AzureChatOpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

from ..tools.meeting_tools import MeetingAssistantTools
from ..core.vector_database import VectorDatabaseManager
from ..config import get_config, PromptTemplates


class MeetingAssistantAgent:
    """Main meeting assistant agent"""
    
    def __init__(self, df_user, azure_config: Dict[str, str] = None):
        self.df_user = df_user
        
        # Get configuration
        config = get_config()
        self.azure_config = azure_config or config.get_azure_config()
        
        self.vector_db = VectorDatabaseManager(df_user)
        self.tools_manager = MeetingAssistantTools(df_user, self.vector_db)
        
        # Initialize LLM
        self.llm = AzureChatOpenAI(
            azure_deployment=self.azure_config["deployment_name"],
            openai_api_version=self.azure_config["api_version"],
            azure_endpoint=self.azure_config["endpoint"],
            api_key=self.azure_config["api_key"],
            temperature=0.1
        )
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Get tools
        self.tools = self.tools_manager.get_tools(self.llm)
        
        # Create agent
        self.agent = self._create_agent()
        
        # Context tracking
        self.current_context = {}
    
    def _create_agent(self):
        """Create the agent with proper prompt"""
        
        system_prompt = PromptTemplates.SYSTEM_PROMPT

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=15,
            memory=self.memory
        )
    
    def run(self, user_input: str) -> Dict[str, Any]:
        """Process user meeting request - main entry point"""
        try:
            print(f"Processing: {user_input}")
            
            result = self.agent.invoke({"input": user_input})
            output = result.get("output", "")
            
            # Try to extract JSON from output
            json_result = self._extract_json_from_output(output)
            
            if json_result:
                return json_result
            else:
                return {"response": output}
                
        except Exception as e:
            return {"error": f"Processing error: {str(e)}"}
    
    def _extract_json_from_output(self, output: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from agent output"""
        try:
            # Look for JSON patterns
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(json_pattern, output, re.DOTALL)
            
            for match in matches:
                try:
                    parsed = json.loads(match)
                    # Check if it's a valid meeting JSON
                    if "subject" in parsed and ("startTime" in parsed or "user_details" in parsed):
                        return parsed
                except:
                    continue
            
            return None
        except:
            return None
    
    def handle_clarification(self, clarification: str) -> Dict[str, Any]:
        """Handle user clarification response"""
        clarification_prompt = PromptTemplates.CLARIFICATION_HANDLING_PROMPT.format(
            clarification=clarification
        )
        
        return self.run(clarification_prompt) 