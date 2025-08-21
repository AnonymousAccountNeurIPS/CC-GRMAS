import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from config import Config
from api_utils import make_api_request, display_response

def show_risk_distribution(base_url: str):
    st.header("Risk Distribution Analysis")
    
    distance_threshold = st.slider(
        "Analysis Distance Threshold (km)",
        min_value=50.0,
        max_value=300.0,
        value=100.0,
        key="risk_distribution_distance"
    )
    
    if st.button("Analyze Risk Distribution", key="risk_distribution_button"):
        with st.spinner("Analyzing current risk distribution..."):
            response = make_api_request(
                base_url,
                Config.ENDPOINTS["risk_distribution"],
                "POST",
                {"distance_threshold": distance_threshold}
            )
            data = display_response(response, "Risk distribution analysis completed!")
            
            if data:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Events Analyzed", data.get("total_analyzed", 0))
                
                with col2:
                    high_risk_pct = data.get("high_risk_percentage", 0)
                    st.metric("High Risk Events", f"{high_risk_pct:.1f}%")
                
                with col3:
                    risk_dist = data.get("risk_distribution", {})
                    high_risk_count = risk_dist.get("High Risk", 0)
                    st.metric("High Risk Count", high_risk_count)
                
                with col4:
                    medium_risk_count = risk_dist.get("Medium Risk", 0)
                    st.metric("Medium Risk Count", medium_risk_count)
                
                if data.get("risk_distribution"):
                    risk_dist = data["risk_distribution"]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_pie = px.pie(
                            values=list(risk_dist.values()),
                            names=list(risk_dist.keys()),
                            title="Overall Risk Distribution"
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with col2:
                        fig_bar = px.bar(
                            x=list(risk_dist.keys()),
                            y=list(risk_dist.values()),
                            title="Risk Level Counts"
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
                
                if data.get("country_risk_breakdown"):
                    st.subheader("Country-wise Risk Breakdown")
                    country_data = data["country_risk_breakdown"]
                    
                    country_rows = []
                    for country, risks in country_data.items():
                        country_rows.append({
                            "Country": country,
                            "Low Risk": risks.get("Low Risk", 0),
                            "Medium Risk": risks.get("Medium Risk", 0),
                            "High Risk": risks.get("High Risk", 0),
                            "Total": sum(risks.values())
                        })
                    
                    country_df = pd.DataFrame(country_rows)
                    
                    if not country_df.empty:
                        country_df = country_df.sort_values("Total", ascending=False)
                        
                        fig_country = go.Figure()
                        
                        fig_country.add_trace(go.Bar(
                            name='Low Risk',
                            x=country_df['Country'],
                            y=country_df['Low Risk']
                        ))
                        
                        fig_country.add_trace(go.Bar(
                            name='Medium Risk',
                            x=country_df['Country'],
                            y=country_df['Medium Risk']
                        ))
                        
                        fig_country.add_trace(go.Bar(
                            name='High Risk',
                            x=country_df['Country'],
                            y=country_df['High Risk']
                        ))
                        
                        fig_country.update_layout(
                            barmode='stack',
                            title='Risk Distribution by Country',
                            xaxis_title='Country',
                            yaxis_title='Number of Events',
                            template='plotly_white'
                        )
                        
                        st.plotly_chart(fig_country, use_container_width=True)
                        
                        st.dataframe(country_df, use_container_width=True)