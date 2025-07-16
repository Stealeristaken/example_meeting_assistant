"""
Vector database manager for name resolution
"""

import uuid
import pandas as pd
from typing import Dict, List, Any
import chromadb
from sentence_transformers import SentenceTransformer

from ..config import get_config


class VectorDatabaseManager:
    """Manages vector database operations for name resolution"""
    
    def __init__(self, df_user: pd.DataFrame):
        self.df_user = df_user
        
        # Get configuration
        config = get_config()
        self.model = SentenceTransformer(config.vector_db_model)
        
        # Setup ChromaDB
        self.client = chromadb.Client()
        try:
            self.client.delete_collection("users")
        except:
            pass
        self.collection = self.client.create_collection("users")
        
        self._index_users()
        print(f"Vector database initialized with {len(df_user)} users")
    
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
    
    def search_names(self, input_names: List[str], threshold: float = None) -> Dict[str, Any]:
        """Search for names using vector similarity"""
        # Get configuration
        config = get_config()
        if threshold is None:
            threshold = config.vector_db_similarity_threshold
            
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