"""
Meeting data models using Pydantic
"""

from typing import Dict, List, Any
from pydantic import BaseModel, Field


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