"""
Configuration management for the meeting assistant
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for the meeting assistant"""
    
    def __init__(self):
        # Azure OpenAI Configuration
        self.azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        self.azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
        self.azure_deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4o')
        self.azure_api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-12-01-preview')
        
        # Business Rules
        self.business_hours_start = os.getenv('BUSINESS_HOURS_START', '09:00')
        self.business_hours_end = os.getenv('BUSINESS_HOURS_END', '17:00')
        self.default_meeting_duration = int(os.getenv('DEFAULT_MEETING_DURATION', '30'))
        self.default_timezone = os.getenv('DEFAULT_TIMEZONE', 'Europe/Istanbul')
        self.default_language = os.getenv('DEFAULT_LANGUAGE', 'tr')
        
        # Vector Database Settings
        self.vector_db_similarity_threshold = float(os.getenv('VECTOR_DB_SIMILARITY_THRESHOLD', '0.7'))
        self.vector_db_model = os.getenv('VECTOR_DB_MODEL', 'all-MiniLM-L6-v2')
    
    def get_azure_config(self) -> Dict[str, str]:
        """Get Azure OpenAI configuration as dictionary"""
        return {
            "endpoint": self.azure_endpoint,
            "api_key": self.azure_api_key,
            "deployment_name": self.azure_deployment_name,
            "api_version": self.azure_api_version
        }
    
    def validate(self) -> bool:
        """Validate configuration"""
        required_fields = [
            self.azure_endpoint,
            self.azure_api_key,
            self.azure_deployment_name,
            self.azure_api_version
        ]
        
        return all(field is not None and field.strip() != '' for field in required_fields)
    
    def get_business_hours(self) -> Dict[str, str]:
        """Get business hours configuration"""
        return {
            "start": self.business_hours_start,
            "end": self.business_hours_end
        }


# Global config instance
_config = None


def get_config() -> Config:
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = Config()
        if not _config.validate():
            raise ValueError("Invalid configuration. Please check your .env file.")
    return _config 