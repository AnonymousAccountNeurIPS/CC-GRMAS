from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from ccgrmas.constants.grmas import driver
from ccgrmas.constants.config import LangChainConfig

class LangChainAnalyticsService:
    """Advanced analytics service using LangChain for landslide data analysis"""
    
    def __init__(self):
        self.config = LangChainConfig()
        self.driver = driver
        self.llm = ChatGoogleGenerativeAI(
            model=self.config.llm_model,
            google_api_key=self.config.google_api_key,
            temperature=0.2
        )
    
    def analyze_risk_patterns(self, country: str = None, trigger: str = None) -> Dict[str, Any]:
        """Analyze landslide risk patterns with AI insights"""
        try:
            cypher_query = """
            MATCH (e:Event)-[:HAS_PROFILE]->(p:LandslideProfile)
            MATCH (e)-[:LOCATED_NEAR]->(g:GazetteerPoint)
            """
            
            params = {}
            conditions = []
            
            if country:
                conditions.append("g.country_name = $country")
                params['country'] = country
            
            if trigger:
                conditions.append("p.landslide_trigger = $trigger")
                params['trigger'] = trigger
            
            if conditions:
                cypher_query += " WHERE " + " AND ".join(conditions)
            
            cypher_query += """
            RETURN 
                count(e) as total_events,
                sum(e.fatality_count) as total_fatalities,
                sum(e.injury_count) as total_injuries,
                collect(DISTINCT p.landslide_category) as categories,
                collect(DISTINCT p.landslide_trigger) as triggers,
                collect(DISTINCT p.landslide_size) as sizes,
                collect(DISTINCT g.country_name) as countries,
                collect(DISTINCT substring(e.event_date, 5, 2)) as months
            """
            
            with self.driver.session() as session:
                result = session.run(cypher_query, params).single()
                
                if not result:
                    return {"success": False, "message": "No data found for analysis"}
                
                analysis_prompt = PromptTemplate(
                    input_variables=["data_summary"],
                    template="""
                    Analyze the following landslide risk data and provide insights:
                    
                    Data Summary:
                    {data_summary}
                    
                    Please provide a comprehensive risk analysis including:
                    
                    1. Overall Risk Assessment:
                       - Total event frequency and severity
                       - Casualty patterns and trends
                       - High-risk categories and triggers
                    
                    2. Geographic Risk Distribution:
                       - Country-specific risk levels
                       - Regional vulnerability patterns
                       - Geographic clustering insights
                    
                    3. Temporal Risk Patterns:
                       - Seasonal variations and peak months
                       - Temporal trends and climate correlations
                       - Early warning indicators
                    
                    4. Risk Mitigation Recommendations:
                       - Priority areas for intervention
                       - Monitoring and early warning strategies
                       - Infrastructure and policy recommendations
                    
                    5. Climate Change Implications:
                       - How changing climate might affect these patterns
                       - Future risk projections
                       - Adaptation strategies needed
                    
                    Base your analysis strictly on the provided data summary.
                    
                    Analysis:
                    """
                )
                
                data_summary = f"""
                Total Events: {result['total_events']}
                Total Fatalities: {result['total_fatalities']}
                Total Injuries: {result['total_injuries']}
                Landslide Categories: {', '.join(result['categories']) if result['categories'] else 'None'}
                Triggers: {', '.join(result['triggers']) if result['triggers'] else 'None'}
                Sizes: {', '.join(result['sizes']) if result['sizes'] else 'None'}
                Countries: {', '.join(result['countries']) if result['countries'] else 'None'}
                Active Months: {', '.join(result['months']) if result['months'] else 'None'}
                """
                
                analysis_chain = LLMChain(llm=self.llm, prompt=analysis_prompt)
                analysis_result = analysis_chain.invoke({"data_summary": data_summary})["text"]
                
                return {
                    "success": True,
                    "analysis": analysis_result,
                    "data_summary": {
                        "total_events": result['total_events'],
                        "total_fatalities": result['total_fatalities'],
                        "total_injuries": result['total_injuries'],
                        "categories": result['categories'],
                        "triggers": result['triggers'],
                        "sizes": result['sizes'],
                        "countries": result['countries'],
                        "active_months": result['months']
                    }
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Risk pattern analysis failed: {str(e)}"
            }
    
    def generate_climate_impact_report(self, region: str = None) -> Dict[str, Any]:
        """Generate climate impact report for landslide risks"""
        try:
            climate_query = """
            MATCH (e:Event)-[:HAS_PROFILE]->(p:LandslideProfile)
            MATCH (e)-[:LOCATED_NEAR]->(g:GazetteerPoint)
            """
            
            params = {}
            if region:
                climate_query += " WHERE g.admin_division_name = $region OR g.country_name = $region or g.gazetteer_closest_point = $region"
                params['region'] = region
            
            climate_query += """
            RETURN 
                p.landslide_trigger as trigger,
                count(e) as event_count,
                avg(e.fatality_count) as avg_fatalities,
                avg(e.injury_count) as avg_injuries,
                collect(DISTINCT substring(e.event_date, 0, 4)) as years,
                collect(DISTINCT g.country_name) as affected_countries
            ORDER BY event_count DESC
            """
            
            with self.driver.session() as session:
                results = session.run(climate_query, params).data()
                
                if not results:
                    return {"success": False, "message": "No climate data found"}
                
                # Create climate impact prompt
                climate_prompt = PromptTemplate(
                    input_variables=["climate_data", "region_filter"],
                    template="""
                    Generate a comprehensive climate change impact report for landslide risks based on the following data:
                    
                    Region: {region_filter}
                    Climate and Trigger Data:
                    {climate_data}
                    
                    Please provide a detailed report covering:
                    
                    1. Climate-Triggered Landslide Analysis:
                       - Most common climate triggers and their frequency
                       - Severity assessment by trigger type
                       - Temporal trends in climate-related events
                    
                    2. Regional Climate Vulnerability:
                       - Geographic distribution of climate impacts
                       - Regional differences in trigger mechanisms
                       - Areas of highest climate-related risk
                    
                    3. Climate Change Projections:
                       - How changing precipitation patterns might affect landslide frequency
                       - Temperature change impacts on slope stability
                       - Extreme weather event correlations
                    
                    4. Adaptation and Mitigation Strategies:
                       - Climate-resilient infrastructure recommendations
                       - Early warning system improvements
                       - Land use planning considerations
                    
                    5. Monitoring and Research Priorities:
                       - Key indicators to track
                       - Data gaps that need addressing
                       - Research questions for future investigation
                    
                    Report:
                    """
                )
                
                climate_data_text = ""
                for result in results:
                    climate_data_text += f"""
                    Trigger: {result['trigger']}
                    Event Count: {result['event_count']}
                    Average Fatalities: {result['avg_fatalities']:.2f}
                    Average Injuries: {result['avg_injuries']:.2f}
                    Years Active: {', '.join(result['years']) if result['years'] else 'Unknown'}
                    Affected Countries: {', '.join(result['affected_countries']) if result['affected_countries'] else 'Unknown'}
                    ---
                    """
                
                climate_chain = LLMChain(llm=self.llm, prompt=climate_prompt)
                report = climate_chain.invoke({
                    "climate_data": climate_data_text,
                    "region_filter": region or "Global"
                })["text"]
                
                return {
                    "success": True,
                    "report": report,
                    "region": region or "Global",
                    "trigger_analysis": results
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Climate impact report generation failed: {str(e)}"
            }