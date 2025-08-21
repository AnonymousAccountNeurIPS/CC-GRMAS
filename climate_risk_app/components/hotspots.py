import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
from config import Config
from api_utils import make_api_request, display_response

def show_hotspots(base_url: str):
    st.header("Spatial Hotspots")

    
    if "hotspots_data" not in st.session_state:
        st.session_state.hotspots_data = None
    if "hotspots_map_key" not in st.session_state:
        st.session_state.hotspots_map_key = 0
    if "hotspots_generated" not in st.session_state:
        st.session_state.hotspots_generated = False

    
    col1, col2 = st.columns(2)
    with col1:
        center_lat = st.number_input(
            "Center Latitude", 
            value=30.0, 
            min_value=25.0, 
            max_value=37.0, 
            key="hotspots_lat"
        )
        center_lon = st.number_input(
            "Center Longitude", 
            value=82.0, 
            min_value=72.0, 
            max_value=92.0, 
            key="hotspots_lon"
        )
    with col2:
        radius = st.number_input(
            "Radius (degrees)", 
            min_value=0.1, 
            max_value=10.0, 
            value=1.0, 
            format="%.2f", 
            key="hotspots_radius"
        )
        grid_resolution = st.number_input(
            "Grid Resolution", 
            min_value=0.05, 
            max_value=0.5, 
            value=0.25, 
            format="%.2f", 
            key="hotspots_resolution"
        )

    if st.button("Find Hotspots", key="hotspots_button"):
        with st.spinner("Analyzing spatial hotspots..."):
            response = make_api_request(
                base_url,
                Config.ENDPOINTS["hotspots"],
                "POST",
                {
                    "latitude": center_lat,
                    "longitude": center_lon,
                    "radius": radius,
                    "grid_resolution": grid_resolution
                }
            )
            data = display_response(response, "Hotspot analysis completed!")
            
            if data and data.get("hotspots"):
                st.session_state.hotspots_data = data
                st.session_state.hotspots_generated = True
                
                st.session_state.hotspots_map_key += 1

    
    if st.session_state.hotspots_data and st.session_state.hotspots_data.get("hotspots"):
        data = st.session_state.hotspots_data
        hotspots = data["hotspots"]

        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Hotspots", len(hotspots))
        with col2:
            high_risk_count = data.get("high_risk_points", 0)
            st.metric("High Risk Points", high_risk_count)
        with col3:
            medium_risk_count = data.get("medium_risk_points", 0)
            st.metric("Medium Risk Points", medium_risk_count)
        with col4:
            coverage = data.get("analysis", {}).get("coverage_area_km2", 0)
            st.metric("Coverage (km²)", f"{coverage:.0f}")

        
        if data.get("risk_summary"):
            risk_summary = data["risk_summary"]
            fig_hotspots = px.bar(
                x=list(risk_summary.keys()),
                y=list(risk_summary.values()),
                title="Hotspot Risk Distribution",
                color=list(risk_summary.keys()),
                color_discrete_map={
                    "Low Risk": "green",
                    "Medium Risk": "orange", 
                    "High Risk": "red"
                }
            )
            fig_hotspots.update_layout(showlegend=False)
            st.plotly_chart(fig_hotspots, use_container_width=True)

        
        st.subheader("Risk Hotspots Map")
        
        
        current_center_lat = st.session_state.get("hotspots_lat", center_lat)
        current_center_lon = st.session_state.get("hotspots_lon", center_lon)
        
        m = folium.Map(
            location=[current_center_lat, current_center_lon], 
            zoom_start=8, 
            tiles="cartodbpositron"
        )

        
        folium.Circle(
            location=[current_center_lat, current_center_lon],
            radius=radius * 111000,  
            color="blue",
            weight=2,
            fill=False,
            popup="Analysis Area"
        ).add_to(m)

        colors = {"Low Risk": "green", "Medium Risk": "orange", "High Risk": "red"}
        
        
        for hotspot in hotspots:
            try:
                folium.CircleMarker(
                    location=[hotspot["latitude"], hotspot["longitude"]],
                    radius=max(5, min(20, hotspot["risk_score"] * 15)),
                    popup=folium.Popup(f"""
                    <div style="width: 200px;">
                        <b>Risk Hotspot</b><br>
                        Level: {hotspot['risk_level']}<br>
                        Score: {hotspot['risk_score']:.3f}<br>
                        Confidence: {hotspot['confidence']:.3f}<br>
                        Coordinates: ({hotspot['latitude']:.3f}, {hotspot['longitude']:.3f})
                    </div>
                    """, max_width=300),
                    color=colors.get(hotspot["risk_level"], "blue"),
                    fillColor=colors.get(hotspot["risk_level"], "blue"),
                    fillOpacity=0.8,
                    weight=2
                ).add_to(m)
            except (KeyError, TypeError) as e:
                st.warning(f"Skipping invalid hotspot data: {e}")
                continue

        
        map_data = st_folium(
            m, 
            width=700, 
            height=500, 
            key=f"hotspots_map_{st.session_state.hotspots_map_key}",
            returned_objects=["last_object_clicked_popup"]
        )

        
        if map_data.get("last_object_clicked_popup"):
            st.info(f"Clicked on: {map_data['last_object_clicked_popup']}")

        
        st.subheader("Hotspot Details")
        
        
        col1, col2 = st.columns(2)
        with col1:
            risk_level_filter = st.multiselect(
                "Filter by Risk Level",
                options=list(set([h["risk_level"] for h in hotspots])),
                default=list(set([h["risk_level"] for h in hotspots])),
                key="hotspots_risk_filter"
            )
        with col2:
            min_confidence = st.slider(
                "Minimum Confidence",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1,
                key="hotspots_confidence_filter"
            )
        
        
        filtered_hotspots = [
            h for h in hotspots 
            if h["risk_level"] in risk_level_filter and h["confidence"] >= min_confidence
        ]
        
        if filtered_hotspots:
            hotspot_df = pd.DataFrame(filtered_hotspots)
            st.dataframe(hotspot_df, use_container_width=True)
            
            
            st.subheader("Filtered Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_risk_score = hotspot_df["risk_score"].mean()
                st.metric("Avg Risk Score", f"{avg_risk_score:.3f}")
            with col2:
                avg_confidence = hotspot_df["confidence"].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.3f}")
            with col3:
                high_risk_filtered = len(hotspot_df[hotspot_df["risk_level"] == "High Risk"])
                st.metric("High Risk (Filtered)", high_risk_filtered)
        else:
            st.warning("No hotspots match the current filter criteria.")

    elif st.session_state.hotspots_generated:
        st.warning("No hotspot data available. Please try the analysis again.")
    else:
        st.info("Configure the parameters above and click 'Find Hotspots' to begin spatial analysis.")