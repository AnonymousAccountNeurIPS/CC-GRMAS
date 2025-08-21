import os
from typing import Dict, Any, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_neo4j import Neo4jVector
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from ccgrmas.constants.grmas import driver, graph
from ccgrmas.constants.config import LangChainConfig
import uuid

class LangChainGraphRAGService:
    """Service for LangChain-based GraphRAG operations on landslide data"""
    
    def __init__(self):
        self.config = LangChainConfig()
        self.driver = driver
        self.graph = graph
        self.embeddings = None
        self.llm = None
        self.vector_store = None
        self.qa_chain = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize LangChain components"""
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=self.config.embedding_model,
                google_api_key=self.config.google_api_key
            )

            self.llm = ChatGoogleGenerativeAI(
                model=self.config.llm_model,
                google_api_key=self.config.google_api_key,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            self.vector_store = Neo4jVector(
                embedding=self.embeddings,
                url=os.getenv("NEO4J_URI"),
                username=os.getenv("NEO4J_USERNAME"),
                password=os.getenv("NEO4J_PASSWORD"),
                index_name=self.config.vector_index_name,
                node_label=self.config.node_label,
                text_node_property=self.config.text_property,
                embedding_node_property=self.config.embedding_property
            )
            
            self._create_qa_chain()
            
        except Exception as e:
            raise Exception(f"Failed to initialize LangChain components: {str(e)}")
    
    def _create_qa_chain(self):
        """Create the QA chain with custom prompt"""
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=self.config.prompt_template
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": self.config.default_top_k}
            ),
            chain_type_kwargs={"prompt": prompt_template},
            return_source_documents=True
        )
    
    def create_vector_index(self) -> Dict[str, Any]:
        """Create vector index in Neo4j for landslide events"""
        try:
            with self.driver.session() as session:
                index_query = f"""
                CREATE VECTOR INDEX {self.config.vector_index_name} IF NOT EXISTS
                FOR (e:Event) ON (e.{self.config.embedding_property})
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: {self.config.embedding_dimension},
                        `vector.similarity_function`: 'cosine'
                    }}
                }}
                """
                session.run(index_query)

                fulltext_query = f"""
                CREATE FULLTEXT INDEX {self.config.fulltext_index_name} IF NOT EXISTS
                FOR (e:Event) ON EACH [e.event_title, e.event_description, e.location_description]
                """
                session.run(fulltext_query)

                profile_fulltext_query = f"""
                CREATE FULLTEXT INDEX {self.config.fulltext_index_name}_profile IF NOT EXISTS
                FOR (p:LandslideProfile) ON EACH [p.landslide_category, p.landslide_trigger, p.landslide_setting]
                """
                session.run(profile_fulltext_query)
                
                gazetteer_fulltext_query = f"""
                CREATE FULLTEXT INDEX {self.config.fulltext_index_name}_gazetteer IF NOT EXISTS
                FOR (g:GazetteerPoint) ON EACH [g.gazetteer_closest_point, g.country_name, g.admin_division_name]
                """
                session.run(gazetteer_fulltext_query)
                
                source_fulltext_query = f"""
                CREATE FULLTEXT INDEX {self.config.fulltext_index_name}_source IF NOT EXISTS
                FOR (s:Source) ON EACH [s.source_name]
                """
                session.run(source_fulltext_query)
                
                return {
                    "success": True,
                    "message": "Vector and fulltext indexes created successfully",
                    "vector_index": self.config.vector_index_name,
                    "fulltext_index": self.config.fulltext_index_name
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to create indexes: {str(e)}"
        }
    
    def populate_vector_store(self) -> Dict[str, Any]:
        """Populate vector store with landslide event data"""
        try:
            batch_size = 50
            total_processed = 0
            
            with self.driver.session() as session:
                count_query = "MATCH (e:Event) RETURN count(e) as total"
                total_events = session.run(count_query).single()['total']
                
                for offset in range(0, total_events, batch_size):
                    events_query = """
                    MATCH (e:Event)
                    OPTIONAL MATCH (e)-[:HAS_PROFILE]->(p:LandslideProfile)
                    OPTIONAL MATCH (e)-[:LOCATED_NEAR]->(g:GazetteerPoint)
                    OPTIONAL MATCH (e)-[:SOURCED_FROM]->(s:Source)
                    RETURN e, p, g, s
                    SKIP $offset LIMIT $batch_size
                    """
                    
                    results = session.run(events_query, {"offset": offset, "batch_size": batch_size}).data()
                    
                    print(f"Processing batch from offset {offset} with {len(results)} records")
                    if not results:
                        break
                    
                    documents = []
                    
                    for record in results:
                        event = record['e']
                        profile = record['p']
                        gazetteer = record['g']
                        source = record['s']

                        text_parts = []
                        metadata = {}
                        
                        import uuid
                        unique_id = str(uuid.uuid4())

                        if event.get('event_title'):
                            text_parts.append(f"Event: {event['event_title']}")
                            metadata['event_title'] = f"{event['event_title']}_{unique_id}"
                        else:
                            metadata['event_title'] = f"event_{unique_id}"
                        
                        if event.get('event_description'):
                            text_parts.append(f"Description: {event['event_description']}")
                        
                        if event.get('event_date'):
                            text_parts.append(f"Date: {event['event_date']}")
                            metadata['event_date'] = event['event_date']
                        
                        if event.get('location_description'):
                            text_parts.append(f"Location: {event['location_description']}")
                        
                        if event.get('fatality_count') is not None:
                            text_parts.append(f"Fatalities: {event['fatality_count']}")
                            metadata['fatality_count'] = event['fatality_count']
                        
                        if event.get('injury_count') is not None:
                            text_parts.append(f"Injuries: {event['injury_count']}")
                            metadata['injury_count'] = event['injury_count']
                        

                        if profile:
                            if profile.get('landslide_category'):
                                text_parts.append(f"Category: {profile['landslide_category']}")
                                metadata['landslide_category'] = profile['landslide_category']
                            
                            if profile.get('landslide_trigger'):
                                text_parts.append(f"Trigger: {profile['landslide_trigger']}")
                                metadata['landslide_trigger'] = profile['landslide_trigger']
                            
                            if profile.get('landslide_size'):
                                text_parts.append(f"Size: {profile['landslide_size']}")
                                metadata['landslide_size'] = profile['landslide_size']
                            
                            if profile.get('landslide_setting'):
                                text_parts.append(f"Setting: {profile['landslide_setting']}")
                                metadata['landslide_setting'] = profile['landslide_setting']
                        
                        if gazetteer:
                            if gazetteer.get('country_name'):
                                text_parts.append(f"Country: {gazetteer['country_name']}")
                                metadata['country_name'] = gazetteer['country_name']
                            
                            if gazetteer.get('admin_division_name'):
                                text_parts.append(f"Admin Division: {gazetteer['admin_division_name']}")
                                metadata['admin_division_name'] = gazetteer['admin_division_name']
                        
                        if source and source.get('source_name'):
                            text_parts.append(f"Source: {source['source_name']}")
                            metadata['source_name'] = source['source_name']

                        if text_parts:
                            content = " | ".join(text_parts)
                            metadata['node_id'] = unique_id
                            
                            doc = Document(
                                page_content=content,
                                metadata=metadata
                            )
                            documents.append(doc)
                    
                    if documents:
                        try:
                            self.vector_store.add_documents(documents)
                            total_processed += len(documents)
                            print(f"Processed batch: {len(documents)} documents (Total: {total_processed})")
                        except Exception as batch_error:
                            print(f"Error processing batch at offset {offset}: {batch_error}")
                            continue
            
            return {
                "success": True,
                "message": f"Successfully populated vector store with {total_processed} documents",
                "processed_count": total_processed
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to populate vector store: {str(e)}"
            }
            
    def similarity_search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Perform similarity search on vector store"""
        try:
            if not self.vector_store:
                raise Exception("Vector store not initialized")
            
            results = self.vector_store.similarity_search(query, k=top_k)
            
            formatted_results = []
            for doc in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
            
            return {
                "success": True,
                "query": query,
                "results": formatted_results,
                "count": len(formatted_results)
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Similarity search failed: {str(e)}"
            }
    
    def generate_answer(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Generate answer using LangChain QA chain"""
        try:
            if not self.qa_chain:
                raise Exception("QA chain not initialized")
            
            self.qa_chain.retriever.search_kwargs["k"] = top_k
            
            result = self.qa_chain.invoke({"query": query})
            
            sources = []
            for doc in result.get("source_documents", []):
                sources.append({
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata
                })
            
            return {
                "success": True,
                "query": query,
                "answer": result["result"],
                "sources": sources,
                "source_count": len(sources)
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Answer generation failed: {str(e)}"
            }
    
    def hybrid_search_with_cypher(self, query: str, cypher_filter: str = None) -> Dict[str, Any]:
        """Perform hybrid search combining vector similarity and Cypher queries"""
        try:
            vector_results = self.similarity_search(query, top_k=3)
            
            if cypher_filter:
                with self.driver.session() as session:
                    cypher_results = session.run(cypher_filter).data()
            else:
                default_cypher = """
                MATCH (e:Event)-[:HAS_PROFILE]->(p:LandslideProfile)
                MATCH (e)-[:LOCATED_NEAR]->(g:GazetteerPoint)
                WHERE toLower(e.event_description) CONTAINS toLower($query)
                   OR toLower(e.location_description) CONTAINS toLower($query)
                   OR toLower(p.landslide_trigger) CONTAINS toLower($query)
                   OR toLower(g.country_name) CONTAINS toLower($query)
                RETURN e, p, g
                LIMIT 5
                """
                
                with self.driver.session() as session:
                    cypher_results = session.run(default_cypher, {"query": query}).data()
            
            return {
                "success": True,
                "query": query,
                "vector_results": vector_results.get("results", []),
                "cypher_results": cypher_results,
                "total_results": len(vector_results.get("results", [])) + len(cypher_results)
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Hybrid search failed: {str(e)}"
            }