#!/usr/bin/env python3
"""
Simple Meeting Assistant with Vector Database
Interactive chat interface to test name resolution

Install first:
uv install pandas pydantic pytz langchain langchain-openai openai chromadb sentence-transformers numpy
"""

import pandas as pd
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any
from pydantic import BaseModel, Field
import pytz

# Vector Database
import chromadb
from sentence_transformers import SentenceTransformer

# LangChain
from langchain.chat_models import AzureChatOpenAI
from langchain.tools import tool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage
from langchain.schema.messages import HumanMessage
from langchain_core.tools import StructuredTool
from langchain.output_parsers import PydanticOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferMemory


# ============================================================================
# SAMPLE DATA WITH AMBIGUOUS NAMES
# ============================================================================

def create_realistic_company_dataframe():
    """Create realistic company data with many ambiguous cases"""
    return pd.DataFrame([
        # Multiple Ahmets
        {"id": 1, "full_name": "Ahmet Yılmaz", "email_address": "ahmet.yilmaz@company.com"},
        {"id": 2, "full_name": "Ahmet Kaya", "email_address": "ahmet.kaya@company.com"},
        {"id": 3, "full_name": "Ahmet Özkan", "email_address": "a.ozkan@company.com"},
        
        # Multiple Alis
        {"id": 4, "full_name": "Ali Şahin", "email_address": "ali.sahin@company.com"},
        {"id": 5, "full_name": "Ali Demir", "email_address": "ali.demir@company.com"},
        {"id": 6, "full_name": "Ali Can Yılmaz", "email_address": "alican.yilmaz@company.com"},
        
        # Şahin variations
        {"id": 7, "full_name": "Mehmet Şahin", "email_address": "mehmet.sahin@company.com"},
        {"id": 8, "full_name": "Şahin Koç", "email_address": "sahin.koc@company.com"},
        {"id": 9, "full_name": "Şahin Nicat, Ph.D", "email_address": "snicat@company.com"},
        
        # Your original data
        {"id": 10, "full_name": "Arda Orçun", "email_address": "arda.orcun@company.com"},
        {"id": 11, "full_name": "Ege Gülünay", "email_address": "ege.gulunay@company.com"},
        
        # More ambiguous cases
        {"id": 12, "full_name": "Özden Gebizli Orkon", "email_address": "ozden.orkon@company.com"},
        {"id": 13, "full_name": "Fatma Özden", "email_address": "fatma.ozden@company.com"},
        
        # Similar sounding names
        {"id": 14, "full_name": "Emre Çelik", "email_address": "emre.celik@company.com"},
        {"id": 15, "full_name": "Emre Çetin", "email_address": "emre.cetin@company.com"},
        
        # Names that could be confused
        {"id": 16, "full_name": "Deniz Kaya", "email_address": "deniz.kaya@company.com"},
        {"id": 17, "full_name": "Deniz Kayahan", "email_address": "deniz.kayahan@company.com"},
        
        # International variations
        {"id": 18, "full_name": "Can Özgür", "email_address": "can.ozgur@company.com"},
        {"id": 19, "full_name": "Can Öztürk", "email_address": "can.ozturk@company.com"},
        
        # Common last names
        {"id": 20, "full_name": "Selin Demir", "email_address": "selin.demir@company.com"},
        {"id": 21, "full_name": "Burak Demir", "email_address": "burak.demir@company.com"},
    ])


# ============================================================================
# MODELS
# ============================================================================

class MeetingInfo(BaseModel):
    attendees: List[str] = Field(description="Katılımcı isimleri")
    duration_minutes: int = Field(description="Süre (dakika)")
    date_description: str = Field(description="Tarih")
    time_description: str = Field(description="Saat")
    subject: str = Field(description="Konu")
    purpose: str = Field(description="Amaç")
    confidence: float = Field(description="Güvenilirlik skoru")


class DateTimeResult(BaseModel):
    start_datetime: str = Field(description="Başlangıç zamanı")
    end_datetime: str = Field(description="Bitiş zamanı")
    is_valid: bool = Field(description="Geçerli mi")
    error_message: str = Field(description="Hata mesajı")


# ============================================================================
# VECTOR NAME RESOLVER
# ============================================================================

class VectorNameResolver:
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
        print(f"✅ {len(df_user)} users indexed in vector database")
    
    def _index_users(self):
        """Create embeddings for user names"""
        texts = []
        metadatas = []
        
        for _, user in self.df_user.iterrows():
            # Create search variations
            full_name = user['full_name'].lower()
            email_prefix = user['email_address'].split('@')[0].lower()
            
            variants = [
                full_name,
                full_name.replace('ç','c').replace('ğ','g').replace('ı','i').replace('ö','o').replace('ş','s').replace('ü','u'),
                full_name.split()[0],  # First name
                full_name.split()[-1] if len(full_name.split()) > 1 else full_name,  # Last name
                email_prefix,
                email_prefix.replace('.', ' ')
            ]
            
            # Add each variant
            for variant in set(variants):  # Remove duplicates
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
    
    def search_names(self, input_names: List[str], threshold: float = 0.7):
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
            unique_users.sort(key=lambda x: x['similarity'], reverse=True)
            
            if len(unique_users) == 1:
                # Auto-resolve
                user = unique_users[0]
                results['resolved_names'].append({
                    'input_name': name,
                    'matched_user': {
                        'id': user['id'],
                        'full_name': user['full_name'],
                        'email_address': user['email_address']
                    },
                    'match_type': 'vector_auto',
                    'similarity_score': user['similarity']
                })
            elif len(unique_users) > 1:
                # Multiple matches
                results['partial_matches'].append({
                    'input_name': name,
                    'candidates': unique_users[:5]  # Top 5
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
    def __init__(self, df_user: pd.DataFrame):
        self.df_user = df_user
        self.name_resolver = VectorNameResolver(df_user)
        self.timezone = pytz.timezone('Europe/Istanbul')
        self.current_time = datetime.now(self.timezone)
    
    def get_tools(self, llm):
        
        @tool
        def extract_meeting_info(user_input: str) -> str:
            """Toplantı bilgilerini çıkar"""
            parser = PydanticOutputParser(pydantic_object=MeetingInfo)
            
            prompt = f"""Toplantı bilgilerini çıkar:

Şu an: {self.current_time.strftime('%d %B %Y, %H:%M')}

{parser.get_format_instructions()}

Giriş: {user_input}"""
            
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content
        
        @tool
        def search_names(attendee_names: str) -> str:
            """Vector database ile isim ara"""
            names = [n.strip() for n in attendee_names.split(',') if n.strip()]
            results = self.name_resolver.search_names(names)
            return json.dumps(results, ensure_ascii=False, indent=2)
        
        @tool
        def check_clarification(search_results: str) -> str:
            """Açıklama gerekli mi kontrol et ve sonraki adımları belirle"""
            try:
                results = json.loads(search_results)
                
                if results.get('needs_clarification'):
                    text = "Lütfen seçim yapın:\n\n"
                    for match in results.get('partial_matches', []):
                        name = match['input_name']
                        text += f"'{name}' için seçenekler:\n"
                        for i, candidate in enumerate(match['candidates'], 1):
                            text += f"  {i}. {candidate['full_name']} ({candidate['email_address']})\n"
                        text += "\n"
                    
                    # Return a special format that the chat loop can detect
                    response = {
                        "clarification_needed": text.strip(),
                        "search_results": results,  # Include full results for preservation
                        "status": "needs_clarification"
                    }
                    return json.dumps(response, ensure_ascii=False)
                else:
                    resolved_count = len(results.get('resolved_names', []))
                    message = f"✅ Tüm isimler çözümlendi ({resolved_count} katılımcı). Şimdi toplantıyı oluştur: parse_datetime, generate_email, create_meeting_json sırasıyla çalıştır."
                    return json.dumps({"proceed": True, "message": message, "status": "proceed"}, ensure_ascii=False)
            except:
                return json.dumps({"error": "Parse error", "status": "error"}, ensure_ascii=False)
        
        @tool
        def create_meeting_json(subject: str, start_time: str, end_time: str, 
                              duration: int, email_body: str, search_results: str) -> str:
            """Final meeting JSON oluştur"""
            try:
                results = json.loads(search_results)
                
                user_details = []
                for resolved in results.get('resolved_names', []):
                    if 'matched_user' in resolved:
                        user = resolved['matched_user']
                        user_details.append({
                            'id': user['id'],
                            'full_name': user['full_name'],
                            'email_address': user['email_address']
                        })
                
                meeting = {
                    "body": email_body,
                    "endTime": end_time,
                    "meeting_duration": duration,
                    "startTime": start_time,
                    "subject": subject,
                    "user_details": user_details
                }
                
                return json.dumps(meeting, ensure_ascii=False, indent=2)
            except Exception as e:
                return json.dumps({"error": str(e)}, ensure_ascii=False)
        
        @tool
        def create_meeting_with_resolved_names(meeting_info_json: str, search_results_json: str, clarification_choice: str) -> str:
            """Çözümlenmiş isimlerle toplantı oluştur - clarification için"""
            try:
                meeting_info = json.loads(meeting_info_json)
                search_results = json.loads(search_results_json)
                
                # Extract meeting details
                subject = meeting_info.get('subject', 'Toplantı')
                purpose = meeting_info.get('purpose', 'toplantı düzenleme')
                duration = meeting_info.get('duration_minutes', 30)
                date_desc = meeting_info.get('date_description', 'yarın')
                time_desc = meeting_info.get('time_description', 'öğleden sonra')
                
                # Get already resolved names from search results
                resolved_names = []
                user_details = []
                
                for resolved in search_results.get('resolved_names', []):
                    user = resolved['matched_user']
                    resolved_names.append(user['full_name'])
                    user_details.append({
                        'id': user['id'],
                        'full_name': user['full_name'],
                        'email_address': user['email_address']
                    })
                
                # Parse user's clarification choice - handle different formats
                clarification_choices = []
                
                # Handle simple name input like "Ali Şahin"
                if not clarification_choice.startswith('{') and not clarification_choice.startswith('['):
                    clarification_choices = [clarification_choice.strip()]
                else:
                    # Handle JSON or complex input
                    try:
                        parsed_choice = json.loads(clarification_choice)
                        if isinstance(parsed_choice, dict):
                            # Extract name from dict format
                            if 'name' in parsed_choice:
                                clarification_choices.append(parsed_choice['name'])
                            elif 'Şahin' in parsed_choice:
                                # Handle nested format
                                for key, value in parsed_choice.items():
                                    if isinstance(value, dict) and 'name' in value:
                                        clarification_choices.append(value['name'])
                        elif isinstance(parsed_choice, list):
                            clarification_choices = parsed_choice
                    except:
                        # Fallback to simple parsing
                        lines = clarification_choice.strip().split('\n')
                        for line in lines:
                            line = line.strip()
                            if line:
                                if ". " in line:
                                    name_part = line.split(". ", 1)[1].split(" (")[0].strip()
                                    clarification_choices.append(name_part)
                                elif " (" in line:
                                    name_part = line.split(" (")[0].strip()
                                    clarification_choices.append(name_part)
                                else:
                                    clarification_choices.append(line)
                
                # Find the selected user from the candidates
                for partial_match in search_results.get('partial_matches', []):
                    input_name = partial_match['input_name']
                    candidates = partial_match['candidates']
                    
                    for choice in clarification_choices:
                        for candidate in candidates:
                            # More flexible matching
                            choice_lower = choice.lower()
                            candidate_lower = candidate['full_name'].lower()
                            
                            if (choice_lower in candidate_lower or 
                                candidate_lower in choice_lower or
                                choice_lower.replace(' ', '') in candidate_lower.replace(' ', '') or
                                candidate_lower.replace(' ', '') in choice_lower.replace(' ', '')):
                                
                                resolved_names.append(candidate['full_name'])
                                user_details.append({
                                    'id': candidate['id'],
                                    'full_name': candidate['full_name'],
                                    'email_address': candidate['email_address']
                                })
                                break
                
                # Parse datetime
                from datetime import timedelta
                days_ahead = 1
                if "yarın" in date_desc.lower():
                    days_ahead = 1
                elif "bugün" in date_desc.lower():
                    days_ahead = 0
                elif "2 hafta" in date_desc.lower():
                    days_ahead = 14
                
                hour = 14
                if "sabah" in time_desc.lower():
                    hour = 9
                elif "öğle" in time_desc.lower():
                    hour = 12
                elif "akşam" in time_desc.lower():
                    hour = 17
                
                base_time = self.current_time + timedelta(days=days_ahead)
                start_time = base_time.replace(hour=hour, minute=0, second=0, microsecond=0)
                end_time = start_time + timedelta(minutes=duration)
                
                # Generate email
                attendees_str = ", ".join(resolved_names)
                email_body = f"""Sayın {attendees_str},

{subject} konulu toplantı için {purpose} amacıyla bir araya gelmek istiyoruz. 

Katılımınızı bekliyoruz.

Saygılarımla."""
                
                # Create final JSON
                meeting = {
                    "body": email_body,
                    "endTime": end_time.isoformat(),
                    "meeting_duration": duration,
                    "startTime": start_time.isoformat(),
                    "subject": subject,
                    "user_details": user_details
                }
                
                return json.dumps(meeting, ensure_ascii=False, indent=2)
                
            except Exception as e:
                return json.dumps({"error": f"Meeting creation failed: {str(e)}"}, ensure_ascii=False)
        
        @tool
        def parse_datetime(date_desc: str, time_desc: str, duration: int) -> str:
            """Tarih/saat parse et - akıllı çevirme"""
            from datetime import timedelta
            
            # Parse date description
            days_ahead = 1  # Default tomorrow
            if "yarın" in date_desc.lower():
                days_ahead = 1
            elif "bugün" in date_desc.lower():
                days_ahead = 0
            elif "hafta" in date_desc.lower():
                if "2 hafta" in date_desc.lower():
                    days_ahead = 14
                else:
                    days_ahead = 7
            elif "pazartesi" in date_desc.lower():
                days_ahead = 1  # Simplified
            elif "salı" in date_desc.lower():
                days_ahead = 2
                
            # Parse time description
            hour = 14  # Default 2 PM
            if "sabah" in time_desc.lower():
                hour = 9
            elif "öğle" in time_desc.lower():
                hour = 12
            elif "akşam" in time_desc.lower():
                hour = 17
            elif ":" in time_desc:
                try:
                    time_part = time_desc.split(":")[0]
                    hour = int(time_part)
                except:
                    hour = 14
            
            base_time = self.current_time + timedelta(days=days_ahead)
            start_time = base_time.replace(hour=hour, minute=0, second=0, microsecond=0)
            end_time = start_time + timedelta(minutes=duration)
            
            return json.dumps({
                "start_datetime": start_time.isoformat(),
                "end_datetime": end_time.isoformat(),
                "is_valid": True,
                "error_message": ""
            }, ensure_ascii=False)
        
        @tool
        def generate_email(subject: str, purpose: str, attendees: str) -> str:
            """E-posta içeriği oluştur - attendees ile gerçek isimler"""
            if not purpose:
                purpose = "toplantı düzenleme"
            if not subject:
                subject = "Toplantı"
            
            return f"""Sayın {attendees},

{subject} konulu toplantı için {purpose} amacıyla bir araya gelmek istiyoruz. 

Katılımınızı bekliyoruz.

Saygılarımla."""
        
        @tool
        def complete_meeting_creation(meeting_info_json: str, search_results_json: str) -> str:
            """Tüm bilgiler hazır olduğunda toplantıyı tamamla"""
            try:
                meeting_info = json.loads(meeting_info_json)
                search_results = json.loads(search_results_json)
                
                # Extract details
                subject = meeting_info.get('subject', 'Toplantı')
                purpose = meeting_info.get('purpose', 'toplantı düzenleme')
                duration = meeting_info.get('duration_minutes', 30)
                date_desc = meeting_info.get('date_description', 'yarın')
                time_desc = meeting_info.get('time_description', 'öğleden sonra')
                
                # Get resolved user names
                attendee_names = []
                for resolved in search_results.get('resolved_names', []):
                    attendee_names.append(resolved['matched_user']['full_name'])
                
                attendees_str = ", ".join(attendee_names)
                
                # Now we have all info - create the meeting step by step
                
                # 1. Parse datetime
                from datetime import timedelta
                days_ahead = 1
                if "yarın" in date_desc.lower():
                    days_ahead = 1
                elif "2 hafta" in date_desc.lower():
                    days_ahead = 14
                
                hour = 14
                if "sabah" in time_desc.lower():
                    hour = 9
                elif "öğle" in time_desc.lower():
                    hour = 12
                
                base_time = self.current_time + timedelta(days=days_ahead)
                start_time = base_time.replace(hour=hour, minute=0, second=0, microsecond=0)
                end_time = start_time + timedelta(minutes=duration)
                
                # 2. Generate email
                email_body = f"""Sayın {attendees_str},

{subject} konulu toplantı için {purpose} amacıyla bir araya gelmek istiyoruz. 

Katılımınızı bekliyoruz.

Saygılarımla."""
                
                # 3. Create final JSON
                user_details = []
                for resolved in search_results.get('resolved_names', []):
                    user = resolved['matched_user']
                    user_details.append({
                        'id': user['id'],
                        'full_name': user['full_name'],
                        'email_address': user['email_address']
                    })
                
                meeting = {
                    "body": email_body,
                    "endTime": end_time.isoformat(),
                    "meeting_duration": duration,
                    "startTime": start_time.isoformat(),
                    "subject": subject,
                    "user_details": user_details
                }
                
                return json.dumps(meeting, ensure_ascii=False, indent=2)
                
            except Exception as e:
                return json.dumps({"error": f"Meeting creation failed: {str(e)}"}, ensure_ascii=False)
        
        return [
            extract_meeting_info,
            search_names,
            check_clarification,
            complete_meeting_creation,
            create_meeting_with_resolved_names,
            parse_datetime,
            generate_email,
            create_meeting_json
        ]


# ============================================================================
# MEETING ASSISTANT AGENT
# ============================================================================

class MeetingAssistantAgent:
    def __init__(self, df_user: pd.DataFrame, azure_config: Dict[str, str]):
        self.df_user = df_user
        
        # Initialize LLM
        self.llm = AzureChatOpenAI(
            azure_endpoint=azure_config["endpoint"],
            api_key=azure_config["api_key"],
            api_version=azure_config["api_version"],
            deployment_name=azure_config["deployment_name"],
            max_tokens=2000
        )
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Tools and agent
        self.tools_manager = MeetingAssistantTools(df_user)
        self.tools = self.tools_manager.get_tools(self.llm)
        self.agent = self._create_agent()
        
        # Store context for clarification
        self.current_context = {}
        
        print("✅ Meeting Assistant ready!")
    
    def _create_agent(self):
        system_prompt = """Sen bir toplantı planlama asistanısın. Vector database ile akıllı isim eşleştirmesi yapıyorsun.

ARAÇLARIN:
- extract_meeting_info: Toplantı bilgilerini çıkar
- search_names: Vector DB ile isim ara
- check_clarification: Belirsizlik kontrol et
- complete_meeting_creation: TEK ADIMDA toplantıyı tamamla (ÖNERİLEN)
- create_meeting_with_resolved_names: Çözümlenmiş isimlerle toplantı oluştur (clarification için)
- parse_datetime: Tarih/saat çevir (alternatif)
- generate_email: E-posta oluştur (alternatif)
- create_meeting_json: Final JSON (alternatif)

ZORUNLU AKIŞ (Bu sırayı takip et):
1. extract_meeting_info ile toplantı bilgilerini çıkar
2. search_names ile katılımcı isimlerini ara
3. check_clarification ile belirsizlik kontrol et
4. EĞER belirsizlik varsa: Kullanıcıdan seçim iste ve DUR
5. EĞER belirsizlik yoksa: 
   SEÇENEK A (ÖNERİLEN): complete_meeting_creation ile tek adımda bitir
   SEÇENEK B: parse_datetime + generate_email + create_meeting_json sırasıyla

KULLANICI SEÇİMİ YAPTIĞINDA:
- Orijinal istekteki TÜM bilgileri koru (tarih, saat, süre, konu)
- Zaten çözümlenmiş isimleri ASLA kaybetme
- create_meeting_with_resolved_names kullanarak tüm isimlerle toplantı oluştur
- Bu tool için 3 parametre gerekli: meeting_info_json, search_results_json, clarification_choice

ÖNEMLİ KURALLAR:
- İsimler çözümlendiyse MUTLAKA toplantıyı tamamla
- complete_meeting_creation en kolay yol - tercih et
- Sadece belirsizlik varsa dur
- Final JSON'ı MUTLAKA döndür - user_details dolu olmalı
- Zaten çözümlenmiş isimleri ASLA kaybetme
- ÖNCEKİ KONUŞMA GEÇMİŞİNİ KULLAN - Memory'deki bilgileri hatırla
- Vector DB'den gelen gerçek kullanıcı bilgilerini kullan

ÖRNEK BAŞARILI AKIŞ:
1. extract_meeting_info ✓
2. search_names ✓  
3. check_clarification → proceed: true ✓
4. complete_meeting_creation → Final JSON ✓

Çözümlenmiş isimlerle MUTLAKA toplantı oluştur!"""

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
            max_iterations=10,
            memory=self.memory
        )
    
    def process_request(self, user_input: str):
        """Process meeting request"""
        try:
            result = self.agent.invoke({"input": user_input})
            output = result.get("output", "")
            
            # Try to extract JSON
            if '{' in output and '}' in output:
                json_start = output.find('{')
                json_end = output.rfind('}') + 1
                json_str = output[json_start:json_end]
                
                try:
                    parsed_result = json.loads(json_str)
                    
                    # Store context for clarification
                    if parsed_result.get("status") == "needs_clarification":
                        self.current_context = {
                            "original_request": user_input,
                            "search_results": parsed_result.get("search_results", {}),
                            "clarification_needed": parsed_result.get("clarification_needed", "")
                        }
                    
                    # Store meeting info if it exists
                    if parsed_result.get("attendees") or parsed_result.get("duration_minutes"):
                        self.last_meeting_info = parsed_result
                    
                    return parsed_result
                except:
                    pass
            
            # If no JSON found, check if the output contains search results
            # The agent might have printed search results in the output
            if "resolved_names" in output or "partial_matches" in output:
                # Try to extract JSON from the output
                try:
                    # Look for JSON patterns in the output
                    import re
                    json_pattern = r'\{[^{}]*"resolved_names"[^{}]*\}'
                    matches = re.findall(json_pattern, output, re.DOTALL)
                    if matches:
                        search_results = json.loads(matches[0])
                        self.current_context = {
                            "original_request": user_input,
                            "search_results": search_results
                        }
                except:
                    pass
            
            return {"response": output}
        except Exception as e:
            return {"error": f"Hata: {str(e)}"}
    
    def handle_clarification(self, original_request: str, clarification: str, original_search_results: dict = None):
        """Handle user clarification - merge with original request properly"""
        
        # Use stored context if available
        meeting_info = None
        if self.current_context:
            original_request = self.current_context.get("original_request", original_request)
            original_search_results = self.current_context.get("search_results", original_search_results)
            # Try to extract meeting info from the context
            if hasattr(self, 'last_meeting_info'):
                meeting_info = self.last_meeting_info
        
        # If we have the original search results, we can be more precise
        if original_search_results:
            # Extract already resolved names
            resolved_names = []
            for resolved in original_search_results.get('resolved_names', []):
                resolved_names.append(resolved['matched_user']['full_name'])
            
            # Parse user's clarification choices - simplified approach
            clarification_choices = []
            
            # Try to extract full names from the clarification
            # User might have entered just numbers, full names, or copied the options
            lines = clarification.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line:
                    # If it's just a number, we'll handle it differently
                    if line.isdigit():
                        choice_num = int(line)
                        # For now, we'll add a placeholder
                        clarification_choices.append(f"Choice {choice_num}")
                    else:
                        # Try to extract a full name from the line
                        # Look for patterns like "Ahmet Yılmaz" or "1. Ahmet Yılmaz"
                        if ". " in line:
                            # Format like "1. Ahmet Yılmaz (email)"
                            name_part = line.split(". ", 1)[1].split(" (")[0].strip()
                            clarification_choices.append(name_part)
                        elif " (" in line:
                            # Format like "Ahmet Yılmaz (email)"
                            name_part = line.split(" (")[0].strip()
                            clarification_choices.append(name_part)
                        else:
                            # Assume it's just a name
                            clarification_choices.append(line)
            
            # Create a comprehensive instruction with all attendees
            all_attendees = resolved_names + clarification_choices
            attendees_str = ", ".join(all_attendees)
            
            clarification_instruction = f"""
Orijinal toplantı isteği: {original_request}

Kullanıcının seçimleri:
- Zaten çözümlenmiş isimler: {', '.join(resolved_names)}
- Kullanıcının seçtiği isimler: {', '.join(clarification_choices)}

Final katılımcı listesi: {attendees_str}

Bu seçimleri kullanarak ORIJINAL toplantı talebini tamamla. Tüm orijinal bilgileri (tarih, saat, süre, konu) koru.
Memory'deki önceki konuşma geçmişini kullan ve orijinal toplantı bilgilerini hatırla.

ÖNEMLİ: create_meeting_with_resolved_names tool'unu kullan ve şu parametreleri ver:
- meeting_info_json: {json.dumps(meeting_info, ensure_ascii=False) if meeting_info else "{}"} (orijinal toplantı bilgileri)
- search_results_json: {json.dumps(original_search_results, ensure_ascii=False)} (orijinal search results)
- clarification_choice: {clarification} (kullanıcının seçimi)

Bu şekilde hem zaten çözümlenmiş isimleri hem de kullanıcının seçtiği isimleri doğru şekilde birleştir.
"""
        else:
            # Fallback to the original approach
            clarification_instruction = f"""
Orijinal toplantı isteği: {original_request}

Kullanıcının belirsiz isimler için yaptığı seçim: {clarification}

ÖNEMLİ: Bu bir seçim, yeni bir istek değil! Orijinal istekteki TÜM bilgileri koru:
- Orijinal tarihi, saati, süreyi, konuyu koru
- Sadece belirsiz olan isimleri kullanıcının seçimiyle değiştir
- Diğer zaten çözümlenmiş isimleri de dahil et
- Memory'deki önceki konuşma geçmişini kullan

Örnek: Eğer orijinalde "Arda ve Şahin" vardı ve kullanıcı "Şahin Nicat" seçtiyse,
final katılımcılar: "Arda Orçun ve Şahin Nicat" olmalı.

create_meeting_with_resolved_names tool'unu kullan ve şu parametreleri ver:
- meeting_info_json: Orijinal toplantı bilgileri
- search_results_json: Orijinal search results
- clarification_choice: Kullanıcının seçimi: {clarification}

Bu seçimi kullanarak ORIJINAL toplantı talebini tamamla.
"""
        
        return self.process_request(clarification_instruction)


# ============================================================================
# INTERACTIVE CHAT
# ============================================================================

def interactive_chat():
    """Full interactive chat with Azure OpenAI"""
    print("🚀 Meeting Assistant Interactive Chat")
    print("=" * 50)
    
    # Load sample data
    df_users = create_realistic_company_dataframe()
    print(f"📊 Loaded {len(df_users)} users")
    print("\nSample users:")
    for _, user in df_users.head(10).iterrows():
        print(f"  • {user['full_name']} ({user['email_address']})")
    print("  ... and more")
    
    # Azure config - load from environment variables
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
        # Initialize the full meeting assistant with Azure OpenAI
        print("\n🤖 Initializing Meeting Assistant with Azure OpenAI...")
        agent = MeetingAssistantAgent(df_users, azure_config)
        
        print("\n" + "="*50)
        print("🎉 FULL CHAT MODE ENABLED!")
        print("You can now have natural conversations about meetings!")
        print("="*50)
        
        print("\n💬 Example prompts to try:")
        print("• 'Arda Orçun ve Şahin ile yarın 90 dakikalık toplantı organize et'")
        print("• 'Ali ve Ahmet ile pazartesi sabah toplantı planla'")
        print("• 'Ozden ve Emre ile proje toplantısı yap'")
        
        # Main chat loop
        while True:
            print("\n" + "-"*50)
            user_input = input("👤 YOU: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'çık', 'bye']:
                break
            
            if not user_input:
                continue
            
            print("🤖 ASSISTANT: Processing your request...")
            
            # Process the request
            result = agent.process_request(user_input)
            
            # Store original search results for clarification
            original_search_results = None
            
            # Check if this is a clarification request
            is_clarification_needed = False
            clarification_text = ""
            
            # Check multiple ways the agent might indicate clarification is needed
            if result.get("status") == "needs_clarification":
                is_clarification_needed = True
                clarification_text = result.get('clarification_needed', '')
                if result.get("search_results"):
                    original_search_results = result["search_results"]
            elif result.get("clarification_needed"):
                is_clarification_needed = True
                clarification_text = result['clarification_needed']
                if result.get("search_results"):
                    original_search_results = result["search_results"]
            elif result.get("response") and "seçim yapın" in result.get("response", ""):
                # The agent returned clarification text in the response
                is_clarification_needed = True
                clarification_text = result.get("response", "")
                # Use stored context if available
                if agent.current_context:
                    original_search_results = agent.current_context.get("search_results")
            
            if is_clarification_needed:
                # Handle clarification
                print(f"\n🤖 ASSISTANT: {clarification_text}")
                
                clarification = input("\n👤 YOUR CHOICE: ").strip()
                
                if clarification:
                    print("🤖 ASSISTANT: Processing your selection...")
                    final_result = agent.handle_clarification(user_input, clarification, original_search_results)
                    
                    if final_result.get("body"):  # Success - meeting created
                        print("\n✅ MEETING CREATED SUCCESSFULLY!")
                        print(json.dumps(final_result, ensure_ascii=False, indent=2))
                        # Clear context after successful meeting creation
                        agent.current_context = {}
                    else:
                        print(f"🤖 ASSISTANT: {final_result.get('response', 'Something went wrong')}")
                
            elif result.get("body"):  # Direct success
                print("\n✅ MEETING CREATED SUCCESSFULLY!")
                print(json.dumps(result, ensure_ascii=False, indent=2))
                # Clear context after successful meeting creation
                agent.current_context = {}
                
            elif result.get("response"):
                print(f"🤖 ASSISTANT: {result['response']}")
                
            elif result.get("error"):
                print(f"❌ ERROR: {result['error']}")
                
            else:
                print("🤖 ASSISTANT: I couldn't process that request. Please try again.")
    
    except Exception as e:
        print(f"❌ Failed to initialize full chat mode: {e}")
        print("\n📋 Falling back to vector name resolution test...")
        
        # Fallback to vector testing only
        resolver = VectorNameResolver(df_users)
        
        while True:
            print("\n" + "-"*30)
            user_input = input("Enter names to search (or 'quit'): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            # Parse names
            names = [n.strip() for n in user_input.split(',') if n.strip()]
            if not names:
                continue
            
            print(f"\nSearching for: {names}")
            results = resolver.search_names(names)
            
            print(f"Needs clarification: {results['needs_clarification']}")
            
            # Show resolved names
            for resolved in results['resolved_names']:
                score = resolved.get('similarity_score', 'N/A')
                print(f"✅ '{resolved['input_name']}' → {resolved['matched_user']['full_name']} (score: {score})")
            
            # Show ambiguous names
            for partial in results['partial_matches']:
                print(f"❓ '{partial['input_name']}' has {len(partial['candidates'])} matches:")
                for i, candidate in enumerate(partial['candidates'], 1):
                    score = candidate.get('similarity', 'N/A')
                    print(f"   {i}. {candidate['full_name']} (score: {score})")
    
    print("\n👋 Thanks for testing!")


# ============================================================================
# MAIN - INTERACTIVE CHAT ONLY
# ============================================================================

def main():
    """Run interactive chat directly"""
    interactive_chat()


if __name__ == "__main__":
    main()