from typing import List, Dict
from langchain.schema import Document

class LangChainHelpers:
    """Helper utilities for LangChain GraphRAG operations"""
    
    @staticmethod
    def format_landslide_document(event_data: Dict, profile_data: Dict = None, 
                                  gazetteer_data: Dict = None, source_data: Dict = None) -> Document:
        """Format landslide data into LangChain Document"""
        
        text_parts = []
        metadata = {"node_type": "Event"}
        
        if event_data.get('event_title'):
            text_parts.append(f"Landslide Event: {event_data['event_title']}")
            metadata['event_title'] = event_data['event_title']
        
        if event_data.get('event_description'):
            text_parts.append(f"Description: {event_data['event_description']}")
        
        if event_data.get('event_date'):
            text_parts.append(f"Date: {event_data['event_date']}")
            metadata['event_date'] = event_data['event_date']
        
        if event_data.get('location_description'):
            text_parts.append(f"Location: {event_data['location_description']}")

        if event_data.get('fatality_count') is not None:
            text_parts.append(f"Fatalities: {event_data['fatality_count']}")
            metadata['fatality_count'] = event_data['fatality_count']
        
        if event_data.get('injury_count') is not None:
            text_parts.append(f"Injuries: {event_data['injury_count']}")
            metadata['injury_count'] = event_data['injury_count']

        if event_data.get('latitude') and event_data.get('longitude'):
            text_parts.append(f"Coordinates: {event_data['latitude']}, {event_data['longitude']}")
            metadata['coordinates'] = f"{event_data['latitude']}, {event_data['longitude']}"

        if profile_data:
            profile_parts = []
            for key in ['landslide_category', 'landslide_trigger', 'landslide_size', 'landslide_setting']:
                if profile_data.get(key):
                    profile_parts.append(f"{key.replace('landslide_', '').title()}: {profile_data[key]}")
                    metadata[key] = profile_data[key]
            
            if profile_parts:
                text_parts.append(f"Profile: {' | '.join(profile_parts)}")
        
        if gazetteer_data:
            geo_parts = []
            if gazetteer_data.get('country_name'):
                geo_parts.append(f"Country: {gazetteer_data['country_name']}")
                metadata['country_name'] = gazetteer_data['country_name']
            
            if gazetteer_data.get('admin_division_name'):
                geo_parts.append(f"Region: {gazetteer_data['admin_division_name']}")
                metadata['admin_division_name'] = gazetteer_data['admin_division_name']
            
            if geo_parts:
                text_parts.append(' | '.join(geo_parts))
        
        if source_data and source_data.get('source_name'):
            text_parts.append(f"Source: {source_data['source_name']}")
            metadata['source_name'] = source_data['source_name']
        
        content = " | ".join(text_parts)
        
        return Document(page_content=content, metadata=metadata)
    
    @staticmethod
    def create_landslide_analysis_prompt() -> str:
        """Create specialized prompt for landslide risk analysis"""
        return """
        You are an expert in landslide risk assessment and climate change analysis.
        Analyze the provided landslide event data to answer questions about:
        
        - Risk patterns and geographic distribution
        - Temporal trends and seasonal variations  
        - Trigger mechanisms and climate relationships
        - Impact assessment and casualty patterns
        - Risk mitigation recommendations
        
        Context: {context}
        
        Question: {question}
        
        Provide a comprehensive analysis with specific references to the data.
        """
    
    @staticmethod
    def extract_key_entities(text: str) -> Dict[str, List[str]]:
        """Extract key entities from landslide text for enhanced search"""
        landslide_triggers = ['rain', 'rainfall', 'downpour', 'earthquake', 'construction', 'erosion', 'flood']
        landslide_types = ['debris flow', 'rock fall', 'mudslide', 'landslide', 'slope failure']
        locations = ['highway', 'road', 'village', 'city', 'mountain', 'hill', 'valley']
        
        text_lower = text.lower()
        
        found_triggers = [trigger for trigger in landslide_triggers if trigger in text_lower]
        found_types = [ltype for ltype in landslide_types if ltype in text_lower]
        found_locations = [loc for loc in locations if loc in text_lower]
        
        return {
            "triggers": found_triggers,
            "types": found_types,
            "locations": found_locations
        }