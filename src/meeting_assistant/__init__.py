"""
Enhanced Meeting Assistant Agent
Fully LLM-driven meeting scheduling with Azure OpenAI and Vector Database
"""

from .core import MeetingAssistantAgent, VectorDatabaseManager
from .models import MeetingRequest, MeetingOutput, NameSearchResult
from .tools import MeetingAssistantTools
from .utils import create_sample_user_data, HealthChecker

__version__ = "1.0.0"
__author__ = "Meeting Assistant Team"

__all__ = [
    'MeetingAssistantAgent',
    'VectorDatabaseManager', 
    'MeetingRequest',
    'MeetingOutput',
    'NameSearchResult',
    'MeetingAssistantTools',
    'create_sample_user_data',
    'HealthChecker'
] 