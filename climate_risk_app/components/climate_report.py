import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config import Config
from api_utils import make_api_request, display_response

def show_climate_report(base_url: str):
    st.header("Climate Impact Report")
    
    region_filter = st.text_input(
        "Focus Region:",
        value="Himachal Pradesh",
        placeholder="e.g., California, Himalayas, Southeast Asia...",
        key="climate_report_region"
    )
    
    if st.button("Generate Climate Report", key="climate_report_button"):
        with st.spinner("Generating climate impact report..."):
            payload = {}
            if region_filter.strip():
                payload["region"] = region_filter.strip()
            
            response = make_api_request(
                base_url,
                Config.ENDPOINTS["climate_report"],
                "POST",
                payload
            )
            data = display_response(response, "Climate report generated!")
            
            if data:
                st.write("### Climate Impact Report")
                st.markdown(data.get("report", "No report available"))
                
                if data.get("trigger_analysis"):
                    trigger_data = data["trigger_analysis"]
                    
                    df_triggers = pd.DataFrame(trigger_data)
                    
                    if not df_triggers.empty:
                        fig = make_subplots(
                            rows=2, cols=2,
                            subplot_titles=("Event Count by Trigger", "Avg Fatalities by Trigger", 
                                          "Avg Injuries by Trigger", "Countries Affected by Trigger")
                        )
                        
                        fig.add_trace(
                            go.Bar(x=df_triggers["trigger"], y=df_triggers["event_count"], name="Event Count"),
                            row=1, col=1
                        )
                        
                        fig.add_trace(
                            go.Bar(x=df_triggers["trigger"], y=df_triggers["avg_fatalities"], name="Avg Fatalities"),
                            row=1, col=2
                        )
                        
                        fig.add_trace(
                            go.Bar(x=df_triggers["trigger"], y=df_triggers["avg_injuries"], name="Avg Injuries"),
                            row=2, col=1
                        )
                        
                        country_counts = [len(countries) if countries else 0 for countries in df_triggers["affected_countries"]]
                        fig.add_trace(
                            go.Bar(x=df_triggers["trigger"], y=country_counts, name="Countries Affected"),
                            row=2, col=2
                        )
                        
                        fig.update_layout(height=600, showlegend=False, title_text="Climate Trigger Analysis")
                        st.plotly_chart(fig, use_container_width=True)