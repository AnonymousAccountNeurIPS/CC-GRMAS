import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from config import Config
import requests

def show_spatial_events(base_url: str):
    st.header("Regional Event Analysis")
    
    
    if "spatial_events_data" not in st.session_state:
        st.session_state.spatial_events_data = None
    if "spatial_events_map_key" not in st.session_state:
        st.session_state.spatial_events_map_key = 0
    if "spatial_events_generated" not in st.session_state:
        st.session_state.spatial_events_generated = False
    
    col1, col2 = st.columns(2)
    
    with col1:
        lat_min = st.number_input("Min Latitude", value=25.0, key="spatial_events_lat_min")
        lat_max = st.number_input("Max Latitude", value=36.0, key="spatial_events_lat_max")
    
    with col2:
        lon_min = st.number_input("Min Longitude", value=72.0, key="spatial_events_lon_min")
        lon_max = st.number_input("Max Longitude", value=91.0, key="spatial_events_lon_max")
    
    
    if lat_min >= lat_max or lon_min >= lon_max:
        st.error("Invalid bounds: minimum values must be less than maximum values.")
        return
    
    if st.button("Analyze Regional Events", key="spatial_events_button"):
        with st.spinner("Fetching regional events..."):
            bounds = {
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max
            }
            
            try:
                url = f"{base_url}{Config.ENDPOINTS['events_spatial']}"
                response = requests.post(url, json=bounds)

                if response.status_code == 200:
                    data = response.json()
                    st.success("Regional events retrieved successfully!")
                    
                    if data.get("events"):
                        
                        st.session_state.spatial_events_data = {
                            "events": data["events"],
                            "bounds": bounds
                        }
                        st.session_state.spatial_events_generated = True
                        
                        st.session_state.spatial_events_map_key += 1
                    else:
                        st.info("No events found in the specified region.")
                        st.session_state.spatial_events_data = None
                        st.session_state.spatial_events_generated = True
                
                else:
                    st.error(f"Error fetching regional events: Status {response.status_code}")
                    st.session_state.spatial_events_data = None
                    
            except requests.exceptions.ConnectionError:
                st.error("Connection error: Unable to connect to the API. Please check if the server is running.")
            except requests.exceptions.RequestException as e:
                st.error(f"Request error: {str(e)}")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")

    
    if st.session_state.spatial_events_data and st.session_state.spatial_events_data.get("events"):
        events = st.session_state.spatial_events_data["events"]
        bounds = st.session_state.spatial_events_data["bounds"]
        
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Events Found", len(events))
        with col2:
            total_fatalities = sum(event.get("fatality_count", 0) for event in events)
            st.metric("Total Fatalities", total_fatalities)
        with col3:
            total_injuries = sum(event.get("injury_count", 0) for event in events)
            st.metric("Total Injuries", total_injuries)
        with col4:
            total_affected = sum(event.get("affected_count", 0) for event in events)
            st.metric("Total Affected", total_affected)
        
        
        events_df = pd.DataFrame(events)
        
        
        if not events_df.empty:
            
            
            if "event_type" in events_df.columns:
                event_types = events_df["event_type"].value_counts()
                if len(event_types) > 0:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Event Types:**")
                        for event_type, count in event_types.items():
                            st.write(f"• {event_type}: {count}")

        
        valid_coords = events_df.dropna(subset=["latitude", "longitude"])
        if len(valid_coords) > 0:
            st.subheader("Regional Events Map")
            
            
            center_lat = (bounds["lat_min"] + bounds["lat_max"]) / 2
            center_lon = (bounds["lon_min"] + bounds["lon_max"]) / 2
            
            m = folium.Map(
                location=[center_lat, center_lon], 
                zoom_start=8, 
                tiles="cartodbpositron"
            )
            
            
            folium.Rectangle(
                bounds=[[bounds["lat_min"], bounds["lon_min"]], [bounds["lat_max"], bounds["lon_max"]]],
                color="blue",
                weight=2,
                fill=False,
                popup="Search Area"
            ).add_to(m)
            
            
            for _, event in valid_coords.iterrows():
                try:
                    fatalities = event.get("fatality_count", 0) or 0
                    injuries = event.get("injury_count", 0) or 0
                    affected = event.get("affected_count", 0) or 0
                    
                    
                    if fatalities > 10:
                        color = "red"
                    elif fatalities > 0 or injuries > 10:
                        color = "orange"
                    elif affected > 100:
                        color = "yellow"
                    else:
                        color = "green"
                    
                    
                    impact_score = fatalities * 3 + injuries + affected * 0.1
                    radius = max(5, min(20, impact_score * 0.5))
                    
                    
                    event_date = event.get("event_date", "Unknown")
                    if pd.notna(event_date) and hasattr(event_date, 'strftime'):
                        event_date = event_date.strftime('%Y-%m-%d')
                    
                    folium.CircleMarker(
                        location=[event["latitude"], event["longitude"]],
                        radius=radius,
                        popup=folium.Popup(f"""
                        <div style="width: 250px;">
                            <b>{event.get('event_title', 'Unknown Event')}</b><br>
                            <hr>
                            <b>Date:</b> {event_date}<br>
                            <b>Type:</b> {event.get('event_type', 'Unknown')}<br>
                            <b>Country:</b> {event.get('country_name', 'Unknown')}<br>
                            <hr>
                            <b>Fatalities:</b> {fatalities}<br>
                            <b>Injuries:</b> {injuries}<br>
                            <b>Affected:</b> {affected}<br>
                            <b>Coordinates:</b> ({event['latitude']:.3f}, {event['longitude']:.3f})
                        </div>
                        """, max_width=300),
                        color=color,
                        fillColor=color,
                        fillOpacity=0.7,
                        weight=2
                    ).add_to(m)
                except (KeyError, TypeError, ValueError) as e:
                    st.warning(f"Skipping invalid event data: {e}")
                    continue
            
            
            map_data = st_folium(
                m, 
                width=700, 
                height=500,
                key=f"spatial_events_map_{st.session_state.spatial_events_map_key}",
                returned_objects=["last_object_clicked_popup"]
            )
            
            
            if map_data.get("last_object_clicked_popup"):
                st.info(f"Clicked on: {map_data['last_object_clicked_popup']}")
        
        else:
            st.warning("No events with valid coordinates found in the specified region.")
        
        
        st.subheader("Regional Events Details")
        
        if not events_df.empty:
            
            col1, col2 = st.columns(2)
        
            
            if "country_name" in events_df.columns:
                with col1:
                    country_filter = st.multiselect(
                        "Filter by Country",
                        options=events_df["country_name"].dropna().unique(),
                        default=events_df["country_name"].dropna().unique(),
                        key="country_filter"
                    )
            else:
                country_filter = None
            
            
            with col2:
                min_fatalities = st.number_input(
                    "Min Fatalities",
                    min_value=0,
                    value=0,
                    key="min_fatalities_filter"
                )
            
            
            filtered_df = events_df.copy()
            
            if country_filter is not None:
                filtered_df = filtered_df[
                    (filtered_df["country_name"].isin(country_filter)) | 
                    (filtered_df["country_name"].isna())
                ]
            
            if min_fatalities > 0:
                filtered_df = filtered_df[
                    filtered_df["fatality_count"].fillna(0) >= min_fatalities
                ]
            
            
            if len(filtered_df) > 0:
                st.write(f"Showing {len(filtered_df)} of {len(events_df)} events")
                st.dataframe(filtered_df, use_container_width=True)
                
                
                if st.button("Export Filtered Data as CSV", key="export_spatial_events"):
                    csv_data = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=f"spatial_events_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("No events match the current filter criteria.")
        
    elif st.session_state.spatial_events_generated:
        st.info("No events found in the specified region. Try adjusting the search bounds.")
    else:
        st.info("Configure the geographic bounds above and click 'Analyze Regional Events' to begin the analysis.")
        
        
        with st.expander("Example Regional Bounds"):
            st.write("**South Asia (India/Pakistan/Bangladesh):**")
            st.write("• Lat: 25.0 to 36.0, Lon: 72.0 to 91.0")
            st.write("**Southeast Asia:**")
            st.write("• Lat: -10.0 to 25.0, Lon: 95.0 to 140.0") 
            st.write("**Europe:**")
            st.write("• Lat: 35.0 to 70.0, Lon: -10.0 to 40.0")
            st.write("**North America:**")
            st.write("• Lat: 25.0 to 70.0, Lon: -170.0 to -50.0")