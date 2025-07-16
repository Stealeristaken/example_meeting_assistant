"""
Tools for the meeting assistant agent
"""

import json
import pandas as pd
from typing import Dict, Any
from datetime import datetime
import pytz
from langchain.tools import tool

from ..core.vector_database import VectorDatabaseManager
from ..config import PromptTemplates, get_config


class MeetingAssistantTools:
    """Tools for the meeting assistant agent"""
    
    def __init__(self, df_user: pd.DataFrame, vector_db: VectorDatabaseManager):
        self.df_user = df_user
        self.vector_db = vector_db
        
        # Get configuration
        config = get_config()
        self.current_date = datetime.now(pytz.timezone(config.default_timezone))
    
    def get_tools(self, llm):
        """Get all tools for the agent"""
        
        @tool
        def extract_meeting_info(user_input: str) -> str:
            """
            Kullanıcının toplantı talebinden bilgileri çıkarır.
            
            Args:
                user_input: Kullanıcının toplantı talebi metni
            
            Returns:
                JSON formatında çıkarılan bilgiler
            """
            prompt = PromptTemplates.EXTRACT_MEETING_INFO_PROMPT.format(
                user_input=user_input,
                current_date=self.current_date.strftime('%d %B %Y, %A')
            )
            
            response = llm.invoke(prompt)
            return response.content
        
        @tool
        def search_user_names(attendee_names: str) -> str:
            """
            Verilen isimleri vektör veritabanında arar ve eşleşmeleri bulur.
            
            Args:
                attendee_names: Aranacak isimler (JSON array string)
            
            Returns:
                Arama sonuçları JSON formatında
            """
            try:
                names = json.loads(attendee_names)
                results = self.vector_db.search_names(names)
                return json.dumps(results, ensure_ascii=False, indent=2)
            except Exception as e:
                return json.dumps({"error": f"İsim arama hatası: {str(e)}"})
        
        @tool
        def parse_datetime(date_desc: str, time_desc: str, duration_minutes: int) -> str:
            """
            Tarih ve saat açıklamalarını ISO 8601 formatına çevirir.
            
            Args:
                date_desc: Tarih açıklaması
                time_desc: Saat açıklaması
                duration_minutes: Toplantı süresi (dakika)
            
            Returns:
                Başlangıç ve bitiş zamanları JSON formatında
            """
            prompt = PromptTemplates.PARSE_DATETIME_PROMPT.format(
                date_desc=date_desc,
                time_desc=time_desc,
                duration_minutes=duration_minutes,
                current_date=self.current_date.strftime('%Y-%m-%d %H:%M:%S')
            )
            
            response = llm.invoke(prompt)
            return response.content
        
        @tool
        def generate_email_body(subject: str, purpose: str, attendees: str) -> str:
            """
            Toplantı davet email gövdesi oluşturur.
            
            Args:
                subject: Toplantı konusu
                purpose: Toplantı amacı
                attendees: Katılımcı isimleri
            
            Returns:
                Email gövdesi metni
            """
            prompt = PromptTemplates.GENERATE_EMAIL_BODY_PROMPT.format(
                subject=subject,
                purpose=purpose,
                attendees=attendees
            )
            
            response = llm.invoke(prompt)
            return response.content.strip()
        
        @tool
        def create_final_meeting_json(subject: str, start_time: str, end_time: str, 
                                    duration: int, email_body: str, user_details: str) -> str:
            """
            Final toplantı JSON'ını oluşturur.
            
            Args:
                subject: Toplantı konusu
                start_time: Başlangıç zamanı (ISO 8601)
                end_time: Bitiş zamanı (ISO 8601)
                duration: Süre (dakika)
                email_body: Email gövdesi
                user_details: Katılımcı detayları (JSON string)
            
            Returns:
                Final toplantı JSON'ı
            """
            try:
                users = json.loads(user_details)
                final_json = {
                    "body": email_body,
                    "endTime": end_time,
                    "meeting_duration": duration,
                    "startTime": start_time,
                    "subject": subject,
                    "user_details": users
                }
                return json.dumps(final_json, ensure_ascii=False, indent=2)
            except Exception as e:
                return json.dumps({"error": f"JSON oluşturma hatası: {str(e)}"})
        
        @tool
        def handle_name_clarification(ambiguous_names: str, candidates: str) -> str:
            """
            Belirsiz isimler için kullanıcıdan açıklama ister.
            
            Args:
                ambiguous_names: Belirsiz isimler listesi
                candidates: Aday kullanıcılar (JSON string)
            
            Returns:
                Açıklama metni
            """
            prompt = PromptTemplates.HANDLE_NAME_CLARIFICATION_PROMPT.format(
                ambiguous_names=ambiguous_names,
                candidates=candidates
            )
            
            response = llm.invoke(prompt)
            return response.content
        
        return [
            extract_meeting_info,
            search_user_names,
            parse_datetime,
            generate_email_body,
            create_final_meeting_json,
            handle_name_clarification
        ] 