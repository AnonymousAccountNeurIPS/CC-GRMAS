import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
from config import Config
from api_utils import make_api_request, display_response

def show_predict_risks(base_url: str):
    st.header("Risk Predictions")

    
    if "predict_risks_data" not in st.session_state:
        st.session_state.predict_risks_data = None
    if "predict_risks_map_key" not in st.session_state:
        st.session_state.predict_risks_map_key = 0
    if "predict_risks_generated" not in st.session_state:
        st.session_state.predict_risks_generated = False

    distance_threshold = st.slider(
        "Distance Threshold (km)",
        min_value=10.0,
        max_value=500.0,
        value=100.0,
        key="predict_risks_distance"
    )

    if st.button("Predict Risks", key="predict_risks_button"):
        with st.spinner("Generating risk predictions..."):
            response = make_api_request(
                base_url,
                Config.ENDPOINTS["predict"],
                "POST",
                {"distance_threshold": distance_threshold}
            )
            data = display_response(response, "Risk predictions generated!")
            
            if data and data.get("predictions"):
                st.session_state.predict_risks_data = data
                st.session_state.predict_risks_generated = True
                
                st.session_state.predict_risks_map_key += 1

    
    if st.session_state.predict_risks_data and st.session_state.predict_risks_data.get("predictions"):
        data = st.session_state.predict_risks_data
        predictions = data["predictions"]

        
        pred_data = []
        for pred in predictions:
            pred_data.append({
                "Event": pred.get("event_title", "Unknown"),
                "Risk Level": pred["prediction"]["risk_level"],
                "Risk Score": pred["prediction"]["risk_score"],
                "Confidence": pred["prediction"]["confidence"],
                "Latitude": pred["location"].get("latitude"),
                "Longitude": pred["location"].get("longitude"),
                "Country": pred["location"].get("country", "Unknown")
            })

        df_pred = pd.DataFrame(pred_data)

        
        risk_counts = df_pred["Risk Level"].value_counts()
        fig_risk = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Risk Level Distribution"
        )
        st.plotly_chart(fig_risk, use_container_width=True)

        
        valid_coords = df_pred.dropna(subset=['Latitude', 'Longitude'])
        if len(valid_coords) > 0:
            st.subheader("Risk Prediction Map")
            
            
            center_lat = valid_coords["Latitude"].mean()
            center_lon = valid_coords["Longitude"].mean()
            
            
            m = folium.Map(
                location=[center_lat, center_lon], 
                zoom_start=6, 
                tiles="cartodbpositron"
            )

            colors = {"Low Risk": "green", "Medium Risk": "orange", "High Risk": "red"}
            
            
            for _, row in valid_coords.iterrows():
                folium.CircleMarker(
                    location=[row["Latitude"], row["Longitude"]],
                    radius=8,
                    popup=folium.Popup(f"""
                    <div style="width: 200px;">
                        <b>{row['Event']}</b><br>
                        Risk: {row['Risk Level']}<br>
                        Score: {row['Risk Score']:.3f}<br>
                        Confidence: {row['Confidence']:.3f}<br>
                        Country: {row['Country']}
                    </div>
                    """, max_width=300),
                    color=colors.get(row["Risk Level"], "blue"),
                    fillColor=colors.get(row["Risk Level"], "blue"),
                    fillOpacity=0.7,
                    weight=2
                ).add_to(m)

            
            map_data = st_folium(
                m, 
                width=700, 
                height=500, 
                key=f"predict_risks_map_{st.session_state.predict_risks_map_key}",
                returned_objects=["last_object_clicked_popup"]
            )
            
            
            if map_data.get("last_object_clicked_popup"):
                st.info(f"Clicked on: {map_data['last_object_clicked_popup']}")

        else:
            st.warning("No valid coordinates found in the prediction data.")

        
        st.subheader("Detailed Predictions")
        
        
        col1, col2 = st.columns(2)
        with col1:
            risk_filter = st.multiselect(
                "Filter by Risk Level",
                options=df_pred["Risk Level"].unique(),
                default=df_pred["Risk Level"].unique(),
                key="risk_level_filter"
            )
        with col2:
            country_filter = st.multiselect(
                "Filter by Country",
                options=df_pred["Country"].unique(),
                default=df_pred["Country"].unique(),
                key="country_filter"
            )
        
        
        filtered_df = df_pred[
            (df_pred["Risk Level"].isin(risk_filter)) & 
            (df_pred["Country"].isin(country_filter))
        ]
        
        st.dataframe(filtered_df, use_container_width=True)
        
        
        if len(filtered_df) > 0:
            st.subheader("Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Predictions", len(filtered_df))
            with col2:
                avg_risk_score = filtered_df["Risk Score"].mean()
                st.metric("Avg Risk Score", f"{avg_risk_score:.3f}")
            with col3:
                avg_confidence = filtered_df["Confidence"].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.3f}")
            with col4:
                high_risk_count = len(filtered_df[filtered_df["Risk Level"] == "High Risk"])
                st.metric("High Risk Events", high_risk_count)
    
    elif st.session_state.predict_risks_generated:
        st.warning("No prediction data available. Please try generating predictions again.")
    else:
        st.info("Click 'Predict Risks' to generate risk predictions for the specified distance threshold.")