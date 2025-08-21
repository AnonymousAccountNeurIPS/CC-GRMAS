import streamlit as st

def show_home_page():

    st.title("🌍 CC-GRMAS Dashboard")
    st.markdown("### Climate Change - Graph Risk Management and Analysis System")
    

    st.markdown("""
    **CC-GRMAS** is an advanced multi-agent framework designed to enhance landslide forecasting and disaster response 
    capabilities in the High Mountain Asia region. Our system leverages cutting-edge graph neural networks, 
    retrieval-augmented generation, and multi-agent coordination to provide real-time situational awareness 
    and proactive disaster preparedness.
    """)
    

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Landslide Events", "1,558", help="Historical events in our database (2007-2020)")
    with col2:
        st.metric("Geographic Coverage", "High Mountain Asia", help="Primary focus region")
    with col3:
        st.metric("Data Sources", "440", help="Information sources and references")
    with col4:
        st.metric("System Agents", "3", help="Prediction, Planning, and Execution agents")
    
    st.markdown("---")
    

    st.markdown("## 🤖 Multi-Agent Architecture")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        
        **Graph Neural Networks**
        - Spatial relationship modeling
        - GCN + GAT architecture  
        - Attention mechanisms
        - Risk classification (Low/Medium/High)
        - Confidence scoring
        """)
    
    with col2:
        st.markdown("""
        
        **Context-Aware Analysis**
        - Large Language Models
        - Retrieval-Augmented Generation
        - Climate impact assessment
        - Risk pattern analysis
        - Domain-specific templates
        """)
    
    with col3:
        st.markdown("""
        
        **Response Coordination**
        - Automated hotspot detection
        - Response generation workflows
        - Spatial risk assessment
        - Grid-based sampling
        - Operational recommendations
        """)
    
    st.markdown("---")
    

    st.markdown("## 📊 Data Foundation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        
        Our system is built upon comprehensive landslide data from the NASA Goddard Space Flight Center:
        
        **Data Sources:**
        - News articles and reports
        - Scientific literature  
        - Government documentation
        - Citizen science contributions
        
        **Temporal Coverage:** 2007-2020
        
        **Graph Database Structure:**
        - Event nodes: Core landslide records
        - Source nodes: Information references
        - GazetteerPoint nodes: Geographic references
        - LandslideProfile nodes: Event characterizations
        """)
    
    with col2:
        st.markdown("""
        
        
        | Type | Count | % |
        |------|-------|---|
        | Event | 1,558 | 61.1% |
        | Source | 440 | 17.2% |
        | GazetteerPoint | 331 | 13.0% |
        | LandslideProfile | 223 | 8.7% |
        | **Total** | **2,552** | **100%** |
        """)
    
    st.markdown("---")

    st.markdown("## 🧭 Navigation Guide")
    
    nav_col1, nav_col2 = st.columns(2)
    
    with nav_col1:
        st.markdown("""
        
        - **Risk Analysis**: Explore spatial risk patterns and trends
        - **Prediction Models**: Access GNN-based forecasting tools
        - **Hotspot Detection**: Identify high-risk geographic areas
        - **Climate Impact**: Generate climate change assessments
        """)
    
    with nav_col2:
        st.markdown("""
        
        - **Graph Database**: Manage landslide event records
        - **Search & Query**: Explore data with AI-powered search
        - **Data Visualization**: Interactive maps and charts
        - **Export Tools**: Download analysis results
        """)
    
    st.markdown("---")
    

    st.markdown("## ✨ Key Features")
    
    features = [
        ("🌐 **Spatial Intelligence**", "Graph-based modeling of landslide spatial dependencies and geographic relationships"),
        ("⚡ **Real-time Processing**", "Dynamic proximity graphs and automated risk assessment workflows"),
        ("🎯 **Multi-scale Analysis**", "From local hotspot detection to regional risk pattern identification"),
        ("🤖 **AI-Powered Insights**", "LLM integration with domain-specific knowledge for contextual analysis"),
        ("📊 **Comprehensive Data**", "Integration of satellite observations, environmental signals, and historical records"),
        ("🚨 **Proactive Response**", "Automated response generation and disaster preparedness recommendations")
    ]
    
    for i, (title, description) in enumerate(features):
        if i % 2 == 0:
            col1, col2 = st.columns(2)
        
        with col1 if i % 2 == 0 else col2:
            st.markdown(f"""
            {title}  
            {description}
            """)
    
    st.markdown("---")
    
    
    st.markdown("## 🚀 Getting Started")
    st.markdown("""
    1. **Explore Data**: Start with the graph database to understand our landslide event records
    2. **Search & Filter**: Use the AI-powered search to find specific events or patterns
    3. **Risk Analysis**: Generate risk assessments for specific regions or time periods
    4. **Predictions**: Access our GNN models for landslide risk forecasting
    5. **Reports**: Generate comprehensive climate impact and response planning reports
    
    Navigate using the sidebar menu to access different system components and begin your analysis.
    """)
    
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
    <p>Powered by Graph Neural Networks • Multi-Agent AI • NASA Global Landslide Catalog</p>
    <p>Tackling Climate Change with Machine Learning • NeurIPS 2025 Workshop</p>
    </div>
    """, unsafe_allow_html=True)