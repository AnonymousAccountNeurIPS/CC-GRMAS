import pandas as pd
from typing import Dict, Any
from ccgrmas.constants.grmas import driver

def create_graph(csv_path: str, clear_existing: bool = True) -> Dict[str, Any]:
    """Create Neo4j graph from CSV data"""

    df = pd.read_csv(csv_path)
    
    if clear_existing:
        with driver.session() as session:
            session.run("MATCH ()-[r]-() DELETE r")
            session.run("MATCH (n) DELETE n")
    
    with driver.session() as session:
        try:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Event) REQUIRE e.event_title IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (g:GazetteerPoint) REQUIRE g.gazetteer_closest_point IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:LandslideProfile) REQUIRE p.profile_id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Source) REQUIRE s.source_name IS UNIQUE")
        except:
            pass
    
    successful_imports = 0
    failed_imports = 0
    
    for _, row in df.iterrows():
        try:
            params = {}

            if pd.notna(row.get('event_title')):
                params['event_title'] = str(row['event_title']).strip()
            if pd.notna(row.get('event_description')):
                params['event_description'] = str(row['event_description']).strip()
            if pd.notna(row.get('event_date')):
                params['event_date'] = str(row['event_date']).strip()
            if pd.notna(row.get('submitted_date')):
                params['submitted_date'] = str(row['submitted_date']).strip()
            if pd.notna(row.get('location_description')):
                params['location_description'] = str(row['location_description']).strip()
            if pd.notna(row.get('location_accuracy')):
                params['location_accuracy'] = str(row['location_accuracy']).strip()
            if pd.notna(row.get('fatality_count')):
                params['fatality_count'] = int(float(row['fatality_count']))
            if pd.notna(row.get('injury_count')):
                params['injury_count'] = int(float(row['injury_count']))
            if pd.notna(row.get('latitude')):
                params['latitude'] = float(row['latitude'])
            if pd.notna(row.get('longitude')):
                params['longitude'] = float(row['longitude'])

            if pd.notna(row.get('gazetteer_closest_point')):
                params['gazetteer_closest_point'] = str(row['gazetteer_closest_point']).strip()
            if pd.notna(row.get('country_name')):
                params['country_name'] = str(row['country_name']).strip()
            if pd.notna(row.get('admin_division_name')):
                params['admin_division_name'] = str(row['admin_division_name']).strip()

            profile_parts = []
            if pd.notna(row.get('landslide_category')):
                params['landslide_category'] = str(row['landslide_category']).strip()
                profile_parts.append(params['landslide_category'])
            if pd.notna(row.get('landslide_trigger')):
                params['landslide_trigger'] = str(row['landslide_trigger']).strip()
                profile_parts.append(params['landslide_trigger'])
            if pd.notna(row.get('landslide_size')):
                params['landslide_size'] = str(row['landslide_size']).strip()
                profile_parts.append(params['landslide_size'])
            if pd.notna(row.get('landslide_setting')):
                params['landslide_setting'] = str(row['landslide_setting']).strip()
                profile_parts.append(params['landslide_setting'])
            
            if profile_parts:
                params['profile_id'] = "_".join(profile_parts)
            
            if pd.notna(row.get('source_name')):
                params['source_name'] = str(row['source_name']).strip()
            if pd.notna(row.get('source_link')):
                params['source_link'] = str(row['source_link']).strip()

            if 'event_title' not in params:
                failed_imports += 1
                continue

            cypher_parts = []
            relationships = []

            event_props = []
            if 'event_title' in params:
                event_props.append("event_title: $event_title")
            if 'event_description' in params:
                event_props.append("event_description: $event_description")
            if 'event_date' in params:
                event_props.append("event_date: $event_date")
            if 'submitted_date' in params:
                event_props.append("submitted_date: $submitted_date")
            if 'location_description' in params:
                event_props.append("location_description: $location_description")
            if 'location_accuracy' in params:
                event_props.append("location_accuracy: $location_accuracy")
            if 'fatality_count' in params:
                event_props.append("fatality_count: $fatality_count")
            if 'injury_count' in params:
                event_props.append("injury_count: $injury_count")
            if 'latitude' in params:
                event_props.append("latitude: $latitude")
            if 'longitude' in params:
                event_props.append("longitude: $longitude")
            
            cypher_parts.append(f"MERGE (e:Event {{event_title: $event_title}}) SET e += {{{', '.join(event_props[1:])}}}")

            if 'gazetteer_closest_point' in params:
                gazetteer_props = ["gazetteer_closest_point: $gazetteer_closest_point"]
                if 'country_name' in params:
                    gazetteer_props.append("country_name: $country_name")
                if 'admin_division_name' in params:
                    gazetteer_props.append("admin_division_name: $admin_division_name")
                
                cypher_parts.append(f"MERGE (g:GazetteerPoint {{gazetteer_closest_point: $gazetteer_closest_point}}) SET g += {{{', '.join(gazetteer_props[1:])}}}")
                relationships.append("MERGE (e)-[:LOCATED_NEAR]->(g)")

            if 'profile_id' in params:
                profile_props = ["profile_id: $profile_id"]
                if 'landslide_category' in params:
                    profile_props.append("landslide_category: $landslide_category")
                if 'landslide_trigger' in params:
                    profile_props.append("landslide_trigger: $landslide_trigger")
                if 'landslide_size' in params:
                    profile_props.append("landslide_size: $landslide_size")
                if 'landslide_setting' in params:
                    profile_props.append("landslide_setting: $landslide_setting")
                
                cypher_parts.append(f"MERGE (p:LandslideProfile {{profile_id: $profile_id}}) SET p += {{{', '.join(profile_props[1:])}}}")
                relationships.append("MERGE (e)-[:HAS_PROFILE]->(p)")

            if 'source_name' in params:
                source_props = ["source_name: $source_name"]
                if 'source_link' in params:
                    source_props.append("source_link: $source_link")
                
                cypher_parts.append(f"MERGE (s:Source {{source_name: $source_name}}) SET s += {{{', '.join(source_props[1:])}}}")
                relationships.append("MERGE (e)-[:SOURCED_FROM]->(s)")
            

            full_query = "\n".join(cypher_parts + relationships)
            

            with driver.session() as session:
                session.run(full_query, params)
            
            successful_imports += 1
            
        except Exception:
            failed_imports += 1
            continue
    
    return {
        "success": True,
        "message": f"Successfully imported {successful_imports} records",
        "records_processed": successful_imports,
        "records_failed": failed_imports,
        "total_records": len(df)
    }