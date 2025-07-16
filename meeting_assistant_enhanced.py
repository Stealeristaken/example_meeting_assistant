#!/usr/bin/env python3
"""
Enhanced Meeting Assistant Agent
Fully LLM-driven meeting scheduling with Azure OpenAI and Vector Database

Requirements:
- All processing through LLM
- Turkish language support
- Vector database for name resolution
- Conversation memory
- Proper JSON output format
"""

import pandas as pd
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import pytz
import re

# Vector Database
import chromadb
from sentence_transformers import SentenceTransformer

# LangChain
from langchain_openai import AzureChatOpenAI
from langchain.tools import tool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage
from langchain_core.tools import StructuredTool
from langchain.output_parsers import PydanticOutputParser
from langchain.memory import ConversationBufferMemory

# ============================================================================
# SAMPLE DATA CREATION
# ============================================================================

def create_sample_user_data():
    """Create realistic sample user data with ambiguous names"""
    return pd.DataFrame([
        # Multiple Ahmets
        {"id": 1, "full_name": "Ahmet Yılmaz", "email_address": "ahmet.yilmaz@company.com.tr"},
        {"id": 2, "full_name": "Ahmet Kaya", "email_address": "ahmet.kaya@company.com.tr"},
        {"id": 3, "full_name": "Ahmet Özkan", "email_address": "a.ozkan@company.com.tr"},
        
        # Multiple Alis
        {"id": 4, "full_name": "Ali Şahin", "email_address": "ali.sahin@company.com.tr"},
        {"id": 5, "full_name": "Ali Demir", "email_address": "ali.demir@company.com.tr"},
        {"id": 6, "full_name": "Ali Can Yılmaz", "email_address": "alican.yilmaz@company.com.tr"},
        
        # Şahin variations
        {"id": 7, "full_name": "Mehmet Şahin", "email_address": "mehmet.sahin@company.com.tr"},
        {"id": 8, "full_name": "Şahin Koç", "email_address": "sahin.koc@company.com.tr"},
        {"id": 9, "full_name": "Şahin Nicat, Ph.D", "email_address": "snicat@company.com.tr"},
        
        # Original data
        {"id": 10, "full_name": "Arda Orçun", "email_address": "arda.orcun@company.com.tr"},
        {"id": 11, "full_name": "Ege Gülünay", "email_address": "ege.gulunay@company.com.tr"},
        
        # More ambiguous cases
        {"id": 12, "full_name": "Özden Gebizli Orkon", "email_address": "ozden.orkon@company.com.tr"},
        {"id": 13, "full_name": "Fatma Özden", "email_address": "fatma.ozden@company.com.tr"},
        
        # Similar sounding names
        {"id": 14, "full_name": "Emre Çelik", "email_address": "emre.celik@company.com.tr"},
        {"id": 15, "full_name": "Emre Çetin", "email_address": "emre.cetin@company.com.tr"},
        
        # Names that could be confused
        {"id": 16, "full_name": "Deniz Kaya", "email_address": "deniz.kaya@company.com.tr"},
        {"id": 17, "full_name": "Deniz Kayahan", "email_address": "deniz.kayahan@company.com.tr"},
        
        # International variations
        {"id": 18, "full_name": "Can Özgür", "email_address": "can.ozgur@company.com.tr"},
        {"id": 19, "full_name": "Can Öztürk", "email_address": "can.ozturk@company.com.tr"},
        
        # Common last names
        {"id": 20, "full_name": "Selin Demir", "email_address": "selin.demir@company.com.tr"},
        {"id": 21, "full_name": "Burak Demir", "email_address": "burak.demir@company.com.tr"},
        
        # Additional users for testing
        {"id": 22, "full_name": "Zeynep Arslan", "email_address": "zeynep.arslan@company.com.tr"},
        {"id": 23, "full_name": "Mert Yıldız", "email_address": "mert.yildiz@company.com.tr"},
        {"id": 24, "full_name": "Elif Özkan", "email_address": "elif.ozkan@company.com.tr"},
        {"id": 25, "full_name": "Kaan Şahin", "email_address": "kaan.sahin@company.com.tr"},
        # New complex ambiguous names
        {"id": 26, "full_name": "Hasan Yıldırım", "email_address": "hasan.yildirim@company.com.tr"},
        {"id": 27, "full_name": "Hasan Can Demir", "email_address": "hasancan.demir@company.com.tr"},
    ])

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class MeetingRequest(BaseModel):
    """Model for extracted meeting information"""
    attendees: List[str] = Field(description="Katılımcı isimleri listesi")
    duration_minutes: int = Field(description="Toplantı süresi (dakika)")
    date_description: str = Field(description="Tarih açıklaması")
    time_description: str = Field(description="Saat açıklaması")
    subject: str = Field(description="Toplantı konusu/başlığı")
    purpose: str = Field(description="Toplantı amacı")
    confidence: float = Field(description="Güvenilirlik skoru (0-1)")

class MeetingOutput(BaseModel):
    """Final meeting output format"""
    body: str = Field(description="Toplantı davet email gövdesi")
    endTime: str = Field(description="ISO 8601 bitiş zamanı")
    meeting_duration: int = Field(description="Toplantı süresi (dakika)")
    startTime: str = Field(description="ISO 8601 başlangıç zamanı")
    subject: str = Field(description="Toplantı konusu")
    user_details: List[Dict[str, Any]] = Field(description="Katılımcı detayları")

class NameSearchResult(BaseModel):
    """Name search result structure"""
    resolved_names: List[Dict[str, Any]] = Field(description="Çözümlenmiş isimler")
    partial_matches: List[Dict[str, Any]] = Field(description="Kısmi eşleşmeler")
    ambiguous_names: List[str] = Field(description="Belirsiz isimler")
    needs_clarification: bool = Field(description="Açıklama gerekli mi")

# ============================================================================
# VECTOR DATABASE MANAGER
# ============================================================================

class VectorDatabaseManager:
    """Manages vector database operations for name resolution"""
    
    def __init__(self, df_user: pd.DataFrame):
        self.df_user = df_user
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Setup ChromaDB
        self.client = chromadb.Client()
        try:
            self.client.delete_collection("users")
        except:
            pass
        self.collection = self.client.create_collection("users")
        
        self._index_users()
        print(f"✅ {len(df_user)} kullanıcı vektör veritabanına eklendi")
    
    def _index_users(self):
        """Create embeddings for user names with variations"""
        texts = []
        metadatas = []
        
        for _, user in self.df_user.iterrows():
            full_name = user['full_name'].lower()
            email_prefix = user['email_address'].split('@')[0].lower()
            
            # Create search variations
            variants = [
                full_name,
                full_name.replace('ç','c').replace('ğ','g').replace('ı','i').replace('ö','o').replace('ş','s').replace('ü','u'),
                full_name.split()[0],  # First name
                full_name.split()[-1] if len(full_name.split()) > 1 else full_name,  # Last name
                email_prefix,
                email_prefix.replace('.', ' '),
                # Add without titles
                full_name.replace(', ph.d', '').replace(', phd', '').strip(),
                # Add common variations
                full_name.replace('şahin', 'sahin').replace('çelik', 'celik')
            ]
            
            # Add each unique variant
            for variant in set(variants):
                if variant.strip():
                    texts.append(variant)
                    metadatas.append({
                        'user_id': user['id'],
                        'full_name': user['full_name'],
                        'email_address': user['email_address']
                    })
        
        # Create embeddings
        embeddings = self.model.encode(texts).tolist()
        ids = [str(uuid.uuid4()) for _ in texts]
        
        self.collection.add(
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts,
            ids=ids
        )
    
    def search_names(self, input_names: List[str], threshold: float = 0.7) -> Dict[str, Any]:
        """Search for names using vector similarity"""
        results = {
            'resolved_names': [],
            'partial_matches': [],
            'ambiguous_names': [],
            'needs_clarification': False
        }
        
        for name in input_names:
            if not name.strip():
                continue
            
            # Vector search
            query_embedding = self.model.encode([name.lower()])
            search_results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=10,
                include=['metadatas', 'distances']
            )
            
            # Group by user
            user_matches = {}
            if search_results['metadatas'][0]:
                for i, metadata in enumerate(search_results['metadatas'][0]):
                    similarity = 1 - search_results['distances'][0][i]
                    if similarity >= threshold:
                        user_id = metadata['user_id']
                        if user_id not in user_matches or similarity > user_matches[user_id]['similarity']:
                            user_matches[user_id] = {
                                'id': user_id,
                                'full_name': metadata['full_name'],
                                'email_address': metadata['email_address'],
                                'similarity': round(similarity, 3)
                            }
            
            unique_users = list(user_matches.values())
            
            if len(unique_users) == 1:
                # Exact match
                results['resolved_names'].append({
                    'input_name': name,
                    'matched_user': unique_users[0],
                    'similarity_score': unique_users[0]['similarity']
                })
            elif len(unique_users) > 1:
                # Multiple matches - needs clarification
                results['partial_matches'].append({
                    'input_name': name,
                    'candidates': unique_users
                })
                results['ambiguous_names'].append(name)
                results['needs_clarification'] = True
            else:
                # No matches
                results['ambiguous_names'].append(name)
                results['needs_clarification'] = True
        
        return results

# ============================================================================
# MEETING ASSISTANT TOOLS
# ============================================================================

class MeetingAssistantTools:
    """Tools for the meeting assistant agent"""
    
    def __init__(self, df_user: pd.DataFrame, vector_db: VectorDatabaseManager):
        self.df_user = df_user
        self.vector_db = vector_db
        self.current_date = datetime.now(pytz.timezone('Europe/Istanbul'))
    
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
            prompt = f"""
            Kullanıcının toplantı talebini analiz et ve aşağıdaki bilgileri çıkar:
            
            Kullanıcı talebi: "{user_input}"
            
            Çıkarılacak bilgiler:
            1. Katılımcılar: Toplantıya katılacak kişilerin isimleri
            2. Süre: Toplantı süresi (dakika cinsinden, belirtilmemişse 30 dakika)
            3. Tarih: Toplantı tarihi (bugün, yarın, pazartesi, vs.)
            4. Saat: Toplantı saati (belirtilmemişse iş saatleri)
            5. Konu: Toplantının konusu/başlığı
            6. Amaç: Toplantının amacı
            
            Mevcut tarih: {self.current_date.strftime('%d %B %Y, %A')}
            
            Lütfen aşağıdaki JSON formatında yanıt ver:
            {{
                "attendees": ["isim1", "isim2"],
                "duration_minutes": 30,
                "date_description": "tarih açıklaması",
                "time_description": "saat açıklaması", 
                "subject": "toplantı konusu",
                "purpose": "toplantı amacı",
                "confidence": 0.9
            }}
            
            Önemli kurallar:
            - Eğer konu belirtilmemişse, "subject" alanını boş bırak
            - Eğer süre belirtilmemişse, 30 dakika varsay
            - Eğer saat belirtilmemişse, iş saatleri (09:00-17:00) varsay
            - Güvenilirlik skorunu 0-1 arasında ver
            """
            
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
            prompt = f"""
            Tarih ve saat bilgilerini ISO 8601 formatına çevir.
            
            Tarih: {date_desc}
            Saat: {time_desc}
            Süre: {duration_minutes} dakika
            Mevcut tarih: {self.current_date.strftime('%Y-%m-%d %H:%M:%S')}
            
            Kurallar:
            - İş saatleri: 09:00-17:00 (UTC+3)
            - Eğer saat belirtilmemişse, 09:00'da başla
            - Toplantı iş saatleri dışına çıkmamalı
            - ISO 8601 formatı: YYYY-MM-DDTHH:MM:SS+03:00
            
            Lütfen aşağıdaki JSON formatında yanıt ver:
            {{
                "startTime": "2025-07-15T09:00:00+03:00",
                "endTime": "2025-07-15T09:30:00+03:00",
                "is_valid": true,
                "error_message": ""
            }}
            """
            
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
            prompt = f"""
            Profesyonel bir toplantı davet email gövdesi oluştur.
            
            Konu: {subject}
            Amaç: {purpose}
            Katılımcılar: {attendees}
            
            Email gövdesi:
            - Kısa ve öz olmalı
            - Profesyonel dil kullan
            - Türkçe yaz
            - Toplantının amacını açıkla
            - Katılımcıları nazikçe davet et
            
            Eğer amaç belirtilmemişse, boş string döndür.
            """
            
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
            prompt = f"""
            Belirsiz isimler için kullanıcıdan açıklama iste.
            
            Belirsiz isimler: {ambiguous_names}
            Aday kullanıcılar: {candidates}
            
            Kullanıcıya nazikçe hangi kişiyi kastettiğini sor.
            Seçenekleri numaralandır.
            Türkçe yaz.
            """
            
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

# ============================================================================
# MEETING ASSISTANT AGENT
# ============================================================================

class MeetingAssistantAgent:
    """Main meeting assistant agent"""
    
    def __init__(self, df_user: pd.DataFrame, azure_config: Dict[str, str]):
        self.df_user = df_user
        self.azure_config = azure_config
        self.vector_db = VectorDatabaseManager(df_user)
        self.tools_manager = MeetingAssistantTools(df_user, self.vector_db)
        
        # Initialize LLM
        self.llm = AzureChatOpenAI(
            azure_deployment=azure_config["deployment_name"],
            openai_api_version=azure_config["api_version"],
            azure_endpoint=azure_config["endpoint"],
            api_key=azure_config["api_key"],
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
        
        system_prompt = """Sen akıllı bir toplantı asistanısın. Kullanıcıların doğal dil toplantı taleplerini işleyerek yapılandırılmış JSON çıktısı oluşturuyorsun.

GÖREV:
1. Kullanıcının toplantı talebini analiz et
2. Katılımcı isimlerini çıkar ve vektör veritabanında ara
3. Tarih/saat bilgilerini ISO 8601 formatına çevir
4. Toplantı konusu ve amacını belirle
5. Email gövdesi oluştur
6. Final JSON çıktısı üret

KURALLAR:
- Tüm işlemleri LLM ile yap, manuel işlem yapma
- Türkçe kullan
- İş saatleri: 09:00-17:00 (UTC+3)
- Varsayılan süre: 30 dakika
- Belirsiz isimler için açıklama iste
- Eksik bilgiler için kullanıcıdan iste

JSON ÇIKTI FORMATI:
{{
    "body": "Email gövdesi",
    "endTime": "2025-07-15T09:30:00+03:00",
    "meeting_duration": 30,
    "startTime": "2025-07-15T09:00:00+03:00", 
    "subject": "Toplantı konusu",
    "user_details": [
        {{
            "email_address": "user@company.com.tr",
            "full_name": "Kullanıcı Adı",
            "id": 123
        }}
    ]
}}

ADIMLAR:
1. extract_meeting_info ile bilgileri çıkar
2. search_user_names ile isimleri ara
3. Eğer belirsizlik varsa handle_name_clarification ile açıklama iste
4. parse_datetime ile tarih/saat çevir
5. generate_email_body ile email oluştur
6. create_final_meeting_json ile final JSON üret

Her adımda Türkçe açıklama ver ve sonuçları kontrol et."""

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
    
    def process_request(self, user_input: str) -> Dict[str, Any]:
        """Process user meeting request"""
        try:
            print(f"🤖 İşleniyor: {user_input}")
            
            result = self.agent.invoke({"input": user_input})
            output = result.get("output", "")
            
            # Try to extract JSON from output
            json_result = self._extract_json_from_output(output)
            
            if json_result:
                return json_result
            else:
                return {"response": output}
                
        except Exception as e:
            return {"error": f"İşlem hatası: {str(e)}"}
    
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
        clarification_prompt = f"""
        Kullanıcı belirsiz isimler için açıklama verdi: {clarification}
        
        Bu açıklamayı kullanarak önceki toplantı talebini tamamla.
        Memory'deki önceki konuşma geçmişini kullan.
        Orijinal toplantı bilgilerini (tarih, saat, süre, konu) koru.
        Sadece belirsiz isimleri kullanıcının seçimiyle değiştir.
        
        Final JSON çıktısını oluştur.
        """
        
        return self.process_request(clarification_prompt)

# ============================================================================
# INTERACTIVE CHAT INTERFACE
# ============================================================================

def interactive_chat():
    """Interactive chat interface"""
    print("🚀 Gelişmiş Toplantı Asistanı")
    print("=" * 50)
    
    # Load sample data
    df_users = create_sample_user_data()
    print(f"📊 {len(df_users)} kullanıcı yüklendi")
    print("\nÖrnek kullanıcılar:")
    for _, user in df_users.head(5).iterrows():
        print(f"  • {user['full_name']} ({user['email_address']})")
    print("  ... ve daha fazlası")
    
    # Azure configuration - load from environment variables
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    azure_config = {
        "endpoint": os.getenv('AZURE_OPENAI_ENDPOINT'),
        "api_key": os.getenv('AZURE_OPENAI_API_KEY'),
        "deployment_name": os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4o'),
        "api_version": os.getenv('AZURE_OPENAI_API_VERSION', '2024-12-01-preview')
    }
    
    # Validate configuration
    if not azure_config["endpoint"] or not azure_config["api_key"]:
        print("❌ ERROR: Azure OpenAI credentials not found!")
        print("Please set the following environment variables:")
        print("  - AZURE_OPENAI_ENDPOINT")
        print("  - AZURE_OPENAI_API_KEY")
        print("  - AZURE_OPENAI_DEPLOYMENT_NAME (optional)")
        print("  - AZURE_OPENAI_API_VERSION (optional)")
        print("\nYou can create a .env file with these variables.")
        return
    
    try:
        # Initialize agent
        print("\n🤖 Toplantı Asistanı başlatılıyor...")
        agent = MeetingAssistantAgent(df_users, azure_config)
        
        print("\n" + "="*50)
        print("🎉 SOHBET MODU AKTİF!")
        print("Doğal dil ile toplantı planlayabilirsiniz!")
        print("="*50)
        
        print("\n💬 Deneyebileceğiniz örnekler:")
        print("• 'Arda Orçun ve Şahin ile yarın 90 dakikalık proje toplantısı organize et'")
        print("• 'Ali ve Ahmet ile pazartesi sabah 10:00'da toplantı planla'")
        print("• 'Ozden ve Emre ile Q3 bütçe görüşmesi yap'")
        print("• 'Yarın saat 14:00'de Arda ile hızlı sync'")
        
        # Main chat loop
        while True:
            print("\n" + "-"*50)
            user_input = input("👤 SİZ: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'çık', 'bye', 'görüşürüz']:
                break
            
            if not user_input:
                continue
            
            print("🤖 ASISTAN: İsteğiniz işleniyor...")
            
            # Process request
            result = agent.process_request(user_input)
            
            # Check if clarification is needed
            if result.get("needs_clarification") or "seçim yapın" in result.get("response", ""):
                print(f"\n🤖 ASISTAN: {result.get('response', 'Açıklama gerekli')}")
                
                clarification = input("\n👤 SEÇİMİNİZ: ").strip()
                
                if clarification:
                    print("🤖 ASISTAN: Seçiminiz işleniyor...")
                    final_result = agent.handle_clarification(clarification)
                    
                    if final_result.get("subject"):
                        print("\n✅ TOPLANTI BAŞARIYLA OLUŞTURULDU!")
                        print(json.dumps(final_result, ensure_ascii=False, indent=2))
                    else:
                        print(f"🤖 ASISTAN: {final_result.get('response', 'Bir hata oluştu')}")
            
            elif result.get("subject"):
                print("\n✅ TOPLANTI BAŞARIYLA OLUŞTURULDU!")
                print(json.dumps(result, ensure_ascii=False, indent=2))
            
            elif result.get("response"):
                print(f"🤖 ASISTAN: {result['response']}")
            
            elif result.get("error"):
                print(f"❌ HATA: {result['error']}")
            
            else:
                print("🤖 ASISTAN: Bu isteği işleyemedim. Lütfen tekrar deneyin.")
    
    except Exception as e:
        print(f"❌ Başlatma hatası: {e}")
        print("\n📋 Vektör veritabanı test moduna geçiliyor...")
        
        # Fallback to vector testing
        vector_db = VectorDatabaseManager(df_users)
        
        while True:
            print("\n" + "-"*30)
            user_input = input("Aranacak isimleri girin (veya 'quit'): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            names = [n.strip() for n in user_input.split(',') if n.strip()]
            if not names:
                continue
            
            print(f"\nAranıyor: {names}")
            results = vector_db.search_names(names)
            
            print(f"Açıklama gerekli: {results['needs_clarification']}")
            
            for resolved in results['resolved_names']:
                score = resolved.get('similarity_score', 'N/A')
                print(f"✅ '{resolved['input_name']}' → {resolved['matched_user']['full_name']} (skor: {score})")
            
            for partial in results['partial_matches']:
                print(f"❓ '{partial['input_name']}' için {len(partial['candidates'])} eşleşme:")
                for i, candidate in enumerate(partial['candidates'], 1):
                    score = candidate.get('similarity', 'N/A')
                    print(f"   {i}. {candidate['full_name']} (skor: {score})")
    
    print("\n👋 Görüşürüz!")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function"""
    interactive_chat()

if __name__ == "__main__":
    main() 