"""
Health check utilities for the meeting assistant
"""

import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer


class HealthChecker:
    """Comprehensive health checker for the meeting assistant system"""
    
    def __init__(self):
        self.checks = []
        self.start_time = time.time()
    
    def check_azure_connection(self, azure_config: Dict[str, str]) -> Dict[str, Any]:
        """Check Azure OpenAI connection"""
        try:
            from langchain_openai import AzureChatOpenAI
            
            llm = AzureChatOpenAI(
                azure_deployment=azure_config["deployment_name"],
                openai_api_version=azure_config["api_version"],
                azure_endpoint=azure_config["endpoint"],
                api_key=azure_config["api_key"],
                temperature=0.1
            )
            
            # Test with a simple prompt
            response = llm.invoke("Merhaba")
            
            return {
                "status": "PASS",
                "message": "Azure OpenAI connection successful",
                "response_time": "OK",
                "details": f"Model: {azure_config['deployment_name']}"
            }
        except Exception as e:
            return {
                "status": "FAIL",
                "message": f"Azure OpenAI connection failed: {str(e)}",
                "response_time": "N/A",
                "details": "Check API key and endpoint configuration"
            }
    
    def check_vector_database(self, df_users: pd.DataFrame) -> Dict[str, Any]:
        """Check vector database functionality"""
        try:
            # Test ChromaDB
            client = chromadb.Client()
            
            # Test collection creation
            try:
                client.delete_collection("health_test")
            except:
                pass
            
            collection = client.create_collection("health_test")
            
            # Test embedding model
            model = SentenceTransformer("all-MiniLM-L6-v2")
            
            # Test with sample data
            sample_texts = ["test", "örnek", "sample"]
            embeddings = model.encode(sample_texts).tolist()
            
            collection.add(
                embeddings=embeddings,
                documents=sample_texts,
                ids=["1", "2", "3"]
            )
            
            # Test search
            query_embedding = model.encode(["test"])
            results = collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=1
            )
            
            # Cleanup
            client.delete_collection("health_test")
            
            return {
                "status": "PASS",
                "message": "Vector database functionality OK",
                "response_time": "OK",
                "details": f"Tested with {len(sample_texts)} samples"
            }
        except Exception as e:
            return {
                "status": "FAIL",
                "message": f"Vector database check failed: {str(e)}",
                "response_time": "N/A",
                "details": "Check ChromaDB and sentence-transformers installation"
            }
    
    def check_user_data(self, df_users: pd.DataFrame) -> Dict[str, Any]:
        """Check user data integrity"""
        try:
            # Check required columns
            required_columns = ['id', 'full_name', 'email_address']
            missing_columns = [col for col in required_columns if col not in df_users.columns]
            
            if missing_columns:
                return {
                    "status": "FAIL",
                    "message": f"Missing required columns: {missing_columns}",
                    "response_time": "N/A",
                    "details": f"Found columns: {list(df_users.columns)}"
                }
            
            # Check for duplicates
            duplicate_emails = df_users[df_users.duplicated(['email_address'], keep=False)]
            duplicate_names = df_users[df_users.duplicated(['full_name'], keep=False)]
            
            # Check for empty values
            empty_emails = df_users[df_users['email_address'].isna() | (df_users['email_address'] == '')]
            empty_names = df_users[df_users['full_name'].isna() | (df_users['full_name'] == '')]
            
            issues = []
            if len(duplicate_emails) > 0:
                issues.append(f"{len(duplicate_emails)} duplicate emails")
            if len(duplicate_names) > 0:
                issues.append(f"{len(duplicate_names)} duplicate names")
            if len(empty_emails) > 0:
                issues.append(f"{len(empty_emails)} empty emails")
            if len(empty_names) > 0:
                issues.append(f"{len(empty_names)} empty names")
            
            status = "PASS" if not issues else "WARN"
            message = "User data integrity OK" if not issues else f"Data issues found: {', '.join(issues)}"
            
            return {
                "status": status,
                "message": message,
                "response_time": "OK",
                "details": f"Total users: {len(df_users)}, Issues: {len(issues)}"
            }
        except Exception as e:
            return {
                "status": "FAIL",
                "message": f"User data check failed: {str(e)}",
                "response_time": "N/A",
                "details": "Check data format and structure"
            }
    
    def check_dependencies(self) -> Dict[str, Any]:
        """Check required dependencies"""
        required_packages = [
            'pandas', 'chromadb', 'sentence_transformers', 
            'langchain_openai', 'langchain', 'pydantic', 'pytz'
        ]
        
        missing_packages = []
        installed_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                installed_packages.append(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            return {
                "status": "FAIL",
                "message": f"Missing packages: {', '.join(missing_packages)}",
                "response_time": "N/A",
                "details": f"Installed: {', '.join(installed_packages)}"
            }
        else:
            return {
                "status": "PASS",
                "message": "All required dependencies installed",
                "response_time": "OK",
                "details": f"Installed packages: {', '.join(installed_packages)}"
            }
    
    def run_full_health_check(self, azure_config: Dict[str, str] = None, df_users: pd.DataFrame = None) -> Dict[str, Any]:
        """Run comprehensive health check"""
        print("Starting comprehensive health check...")
        
        # Get configuration if not provided
        if azure_config is None:
            from ..config import get_config
            config = get_config()
            azure_config = config.get_azure_config()
        
        if df_users is None:
            from ..utils.data_utils import create_sample_user_data
            df_users = create_sample_user_data()
        
        checks = {
            "dependencies": self.check_dependencies(),
            "azure_connection": self.check_azure_connection(azure_config),
            "vector_database": self.check_vector_database(df_users),
            "user_data": self.check_user_data(df_users)
        }
        
        # Calculate overall status
        statuses = [check["status"] for check in checks.values()]
        if "FAIL" in statuses:
            overall_status = "FAIL"
        elif "WARN" in statuses:
            overall_status = "WARN"
        else:
            overall_status = "PASS"
        
        total_time = time.time() - self.start_time
        
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": overall_status,
            "total_checks": len(checks),
            "total_time_seconds": round(total_time, 2),
            "checks": checks,
            "summary": {
                "passed": len([s for s in statuses if s == "PASS"]),
                "warnings": len([s for s in statuses if s == "WARN"]),
                "failed": len([s for s in statuses if s == "FAIL"])
            }
        }
        
        return health_report
    
    def print_health_report(self, report: Dict[str, Any]):
        """Print formatted health report"""
        print("\n" + "="*60)
        print("HEALTH CHECK REPORT")
        print("="*60)
        print(f"Timestamp: {report['timestamp']}")
        print(f"Overall Status: {report['overall_status']}")
        print(f"Total Time: {report['total_time_seconds']} seconds")
        print(f"Checks: {report['summary']['passed']} passed, "
              f"{report['summary']['warnings']} warnings, "
              f"{report['summary']['failed']} failed")
        print("-"*60)
        
        for check_name, check_result in report['checks'].items():
            status_icon = "✅" if check_result['status'] == 'PASS' else "⚠️" if check_result['status'] == 'WARN' else "❌"
            print(f"{status_icon} {check_name.upper()}: {check_result['status']}")
            print(f"   Message: {check_result['message']}")
            print(f"   Details: {check_result['details']}")
            print()
        
        print("="*60)
        
        if report['overall_status'] == 'PASS':
            print("🎉 All systems operational!")
        elif report['overall_status'] == 'WARN':
            print("⚠️  System operational with warnings")
        else:
            print("❌ System has critical issues that need attention")
        
        print("="*60) 