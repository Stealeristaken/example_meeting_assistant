"""
Data utilities for the meeting assistant
"""

import pandas as pd


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