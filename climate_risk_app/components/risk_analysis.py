import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from config import Config
from api_utils import make_api_request, display_response
from collections import Counter
import calendar

def show_risk_analysis(base_url: str):
    st.header("🏔️ Landslide Risk Analysis Dashboard")
    
    
    col1, col2 = st.columns(2)
    with col1:
        country_filter = st.selectbox(
            "🌍 Filter by Country:",
            ["India", "China", "Nepal", "Pakistan", "Bangladesh", "Bhutan"],
            key="risk_analysis_country"
        )
    
    with col2:
        trigger_filter = st.selectbox(
            "🌧️ Filter by Trigger:",
            ["downpour", "continuous_rain", "monsoon", "rain", "unknown"],
            key="risk_analysis_trigger"
        )
    
    if st.button("🔍 Analyze Risk Patterns", key="risk_analysis_button", type="primary"):
        with st.spinner("Analyzing risk patterns..."):
            payload = {}
            if country_filter.strip():
                payload["country"] = country_filter.strip()
            if trigger_filter:
                payload["trigger"] = trigger_filter
            
            response = make_api_request(
                base_url,
                Config.ENDPOINTS["risk_analysis"],
                "POST",
                payload
            )
            data = display_response(response, "Risk analysis completed!")
            
            if data:
                
                display_detailed_analysis(data.get("analysis"))
                
                if data.get("data_summary"):
                    summary = data["data_summary"]
                    
                    
                    display_key_metrics(summary)
                    
                    
                    create_comprehensive_visualizations(summary)

def display_detailed_analysis(analysis_text):
    """Display the detailed analysis report at the top"""
    if analysis_text:
        st.write("## 📋 Comprehensive Risk Analysis Report")
        
        
        formatted_analysis = format_analysis_text(analysis_text)
        
        
        st.markdown(formatted_analysis)
        
        st.divider()

def format_analysis_text(analysis_text):
    """Format analysis text for better markdown display"""
    if not analysis_text:
        return "No analysis available"
    
    
    formatted_text = analysis_text.strip()
    
    
    import re
    
    
    formatted_text = re.sub(r'(\d+\.\s*\*\*([^*]+)\*\*:?)', r'### \2', formatted_text)
    
    
    formatted_text = re.sub(r'^\s*\*\s*\*\*([^*]+)\*\*:', r'**\1:**', formatted_text, flags=re.MULTILINE)
    
    
    formatted_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', formatted_text)
    
    
    formatted_text = re.sub(r'(###[^\n]+)\n([^\n])', r'\1\n\n\2', formatted_text)
    
    return formatted_text

def display_key_metrics(summary):
    """Display key metrics dashboard with dynamic calculations"""
    st.write("## 📊 Key Risk Metrics")
    
    
    total_events = summary.get("total_events", 0)
    total_fatalities = summary.get("total_fatalities", 0)
    total_injuries = summary.get("total_injuries", 0)
    countries_count = len(summary.get("countries", []))
    
    
    total_casualties = total_fatalities + total_injuries
    fatality_rate = round(total_fatalities / total_events if total_events > 0 else 0, 2)
    casualty_rate = round(total_casualties / total_events if total_events > 0 else 0, 2)
    injury_ratio = round(total_injuries / total_fatalities if total_fatalities > 0 else 0, 2)
    
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Events", f"{total_events:,}", help="Total number of landslide events recorded")
    with col2:
        st.metric("Total Fatalities", f"{total_fatalities:,}", help="Total deaths from landslide events")
    with col3:
        st.metric("Total Injuries", f"{total_injuries:,}", help="Total injuries from landslide events")
    with col4:
        st.metric("Fatality Rate", f"{fatality_rate}", help="Average fatalities per event")
    with col5:
        st.metric("Countries Affected", f"{countries_count}", help="Number of countries in the dataset")
    
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Casualties", f"{total_casualties:,}", help="Combined fatalities and injuries")
    with col2:
        st.metric("Casualty Rate", f"{casualty_rate}", help="Average casualties per event")
    with col3:
        st.metric("Injury-to-Fatality Ratio", f"{injury_ratio}", help="Injuries per fatality")
    with col4:
        
        risk_level, risk_color = calculate_risk_level(casualty_rate, fatality_rate)
        st.markdown(f"**Risk Level**: <span style='color: {risk_color}; font-weight: bold;'>{risk_level}</span>", 
                   unsafe_allow_html=True, help="Risk assessment based on casualty rates")

def calculate_risk_level(casualty_rate, fatality_rate):
    """Calculate risk level based on casualty and fatality rates"""
    if casualty_rate >= 10 or fatality_rate >= 8:
        return "🔴 CRITICAL", "red"
    elif casualty_rate >= 5 or fatality_rate >= 3:
        return "🟠 HIGH", "orange"
    elif casualty_rate >= 2 or fatality_rate >= 1:
        return "🟡 MEDIUM", "gold"
    elif casualty_rate > 0:
        return "🟢 LOW", "green"
    else:
        return "⚪ MINIMAL", "gray"

def create_comprehensive_visualizations(summary):
    """Create comprehensive visualizations for the risk analysis data"""
    
    
    create_seasonal_analysis(summary)
    
    
    create_categories_visualization(summary)
    
    
    create_triggers_analysis(summary)
    
    
    create_size_distribution(summary)
    
    
    create_geographic_distribution(summary)
    
    
    create_comparative_analysis(summary)
    
    
    create_risk_summary(summary)

def create_seasonal_analysis(summary):
    """Create seasonal distribution analysis"""
    if not summary.get("active_months"):
        return
        
    st.write("## 📅 Seasonal Risk Patterns")
    
    
    month_names = {str(i).zfill(2): calendar.month_name[i] for i in range(1, 13)}
    month_counts = Counter(summary["active_months"])
    
    
    month_data = []
    for month_code in sorted(month_counts.keys()):
        month_name = month_names.get(month_code, f"Month {month_code}")
        count = month_counts[month_code]
        month_data.append({
            "Month": month_name, 
            "Month_Code": month_code, 
            "Events": count,
            "Month_Num": int(month_code)
        })
    
    month_df = pd.DataFrame(month_data).sort_values("Month_Num")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        
        fig_seasonal = px.bar(
            month_df, 
            x="Month", 
            y="Events",
            title="Landslide Events Distribution by Month",
            color="Events",
            color_continuous_scale="Reds",
            text="Events"
        )
        fig_seasonal.update_layout(height=400, showlegend=False, xaxis_tickangle=45)
        fig_seasonal.update_traces(textposition="outside")
        st.plotly_chart(fig_seasonal, use_container_width=True)
    
    with col2:
        
        st.write("**Seasonal Insights:**")
        
        
        peak_events = max(month_counts.values()) if month_counts else 0
        peak_months = [month_names.get(k, k) for k, v in month_counts.items() if v == peak_events]
        
        if peak_months:
            st.info(f"🔥 **Peak Month(s)**: {', '.join(peak_months)}")
        
        
        monsoon_months = ["06", "07", "08", "09"]
        monsoon_events = sum(month_counts.get(m, 0) for m in monsoon_months)
        total_events = sum(month_counts.values())
        
        if total_events > 0:
            monsoon_percentage = (monsoon_events / total_events) * 100
            st.warning(f"🌧️ **Monsoon Impact**: {monsoon_percentage:.1f}% of events during monsoon season")
        
        
        active_months_count = len([m for m in month_counts.values() if m > 0])
        st.info(f"📊 **Active Months**: {active_months_count}/12 months show activity")

def create_categories_visualization(summary):
    """Create categories distribution visualization"""
    if not summary.get("categories"):
        return
        
    st.write("## 🏔️ Landslide Categories Distribution")
    
    categories = summary["categories"]
    category_counts = Counter(categories) if isinstance(categories, list) else {cat: 1 for cat in categories}
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        
        categories_df = pd.DataFrame([
            {"Category": cat.replace("_", " ").title(), "Count": count} 
            for cat, count in category_counts.items()
        ])
        
        fig_categories = px.pie(
            categories_df, 
            values="Count", 
            names="Category",
            title="Distribution of Landslide Categories"
        )
        fig_categories.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_categories, use_container_width=True)
    
    with col2:
        st.write("**Category Definitions:**")
        category_definitions = get_category_definitions()
        
        for cat in categories:
            clean_cat = cat.replace("_", " ").title()
            definition = category_definitions.get(cat.lower(), "Specific type of mass movement")
            st.write(f"**{clean_cat}**: {definition}")

def get_category_definitions():
    """Get definitions for landslide categories"""
    return {
        "landslide": "General slope failure and mass movement",
        "mudslide": "Flow of water-saturated earth material",
        "debris_flow": "Fast-moving mixture of debris and water",
        "rock_fall": "Falling or bouncing rocks from steep slopes",
        "translational_slide": "Movement along a planar surface",
        "complex": "Multiple types of movement combined",
        "rotational_slide": "Movement along a curved surface",
        "other": "Other types of mass movements",
        "unknown": "Unclassified events"
    }

def create_triggers_analysis(summary):
    """Create triggers analysis visualization"""
    if not summary.get("triggers"):
        return
        
    st.write("## 🌧️ Landslide Triggers Analysis")
    
    triggers = summary["triggers"]
    trigger_counts = Counter(triggers) if isinstance(triggers, list) else {trigger: 1 for trigger in triggers}
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        triggers_df = pd.DataFrame([
            {"Trigger": trigger.replace("_", " ").title(), "Frequency": count}
            for trigger, count in trigger_counts.items()
        ])
        
        fig_triggers = px.bar(
            triggers_df, 
            x="Trigger", 
            y="Frequency",
            title="Landslide Triggers Distribution",
            color="Frequency",
            color_continuous_scale="Blues"
        )
        fig_triggers.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_triggers, use_container_width=True)
    
    with col2:
        st.write("**Trigger Analysis:**")
        
        
        weather_triggers = [t for t in triggers if any(word in t.lower() for word in ['rain', 'monsoon', 'downpour', 'storm', 'snow'])]
        weather_percentage = (len(weather_triggers) / len(triggers)) * 100 if triggers else 0
        
        st.warning(f"🌦️ **Weather Dependency**: {weather_percentage:.0f}% of triggers are weather-related")
        
        
        if trigger_counts:
            most_common = max(trigger_counts.items(), key=lambda x: x[1])
            st.info(f"🔥 **Primary Trigger**: {most_common[0].replace('_', ' ').title()}")

def create_size_distribution(summary):
    """Create size distribution visualization"""
    if not summary.get("sizes"):
        return
        
    st.write("## 📏 Event Size Distribution")
    
    sizes = summary["sizes"]
    size_counts = Counter(sizes) if isinstance(sizes, list) else {size: 1 for size in sizes}
    
    
    size_order = ["very_small", "small", "medium", "large", "very_large", "unknown"]
    ordered_sizes = sorted(size_counts.keys(), key=lambda x: size_order.index(x) if x in size_order else len(size_order))
    
    sizes_df = pd.DataFrame([
        {"Size": size.replace("_", " ").title(), "Count": size_counts[size]}
        for size in ordered_sizes
    ])
    
    
    color_map = {
        "Very Small": "#90EE90", "Small": "#98FB98", "Medium": "#FFD700", 
        "Large": "#FFA500", "Very Large": "#FF6B6B", "Unknown": "#D3D3D3"
    }
    
    fig_sizes = px.bar(
        sizes_df, 
        x="Size", 
        y="Count",
        title="Landslide Event Size Distribution",
        color="Size",
        color_discrete_map=color_map
    )
    fig_sizes.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_sizes, use_container_width=True)

def create_geographic_distribution(summary):
    """Create geographic distribution visualization"""
    if not summary.get("countries"):
        return
        
    st.write("## 🗺️ Geographic Distribution")
    
    countries = summary["countries"]
    country_counts = Counter(countries) if isinstance(countries, list) else {country: 1 for country in countries}
    
    countries_df = pd.DataFrame([
        {"Country": country, "Events": count}
        for country, count in country_counts.items()
    ])
    
    fig_countries = px.bar(
        countries_df, 
        x="Country", 
        y="Events",
        title="Landslide Events by Country",
        color="Events",
        color_continuous_scale="Viridis"
    )
    fig_countries.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_countries, use_container_width=True)

def create_comparative_analysis(summary):
    """Create comparative analysis dashboard"""
    st.write("## 📊 Comparative Risk Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        
        fig_casualties = px.pie(
            values=[summary.get("total_fatalities", 0), summary.get("total_injuries", 0)],
            names=["Fatalities", "Injuries"],
            title="Casualties Distribution",
            color_discrete_map={"Fatalities": "#FF6B6B", "Injuries": "#4ECDC4"}
        )
        st.plotly_chart(fig_casualties, use_container_width=True)
    
    with col2:
        
        total_events = summary.get("total_events", 0)
        total_casualties = summary.get("total_fatalities", 0) + summary.get("total_injuries", 0)
        severity_score = (total_casualties / total_events * 10) if total_events > 0 else 0
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = min(severity_score, 100),
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risk Severity Index"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred" if severity_score > 50 else "orange" if severity_score > 20 else "green"},
                'steps': [
                    {'range': [0, 20], 'color': "lightgray"},
                    {'range': [20, 50], 'color': "gray"},
                    {'range': [50, 100], 'color': "darkgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 75
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

def create_risk_summary(summary):
    """Create risk assessment summary"""
    st.write("## ⚠️ Risk Assessment Summary")
    
    total_events = summary.get("total_events", 0)
    total_casualties = summary.get("total_fatalities", 0) + summary.get("total_injuries", 0)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        casualty_rate = total_casualties / total_events if total_events > 0 else 0
        risk_level, _ = calculate_risk_level(casualty_rate, summary.get("total_fatalities", 0) / total_events if total_events > 0 else 0)
        st.markdown(f"### Overall Risk Level\n#### {risk_level}")
    
    with col2:
        severity_score = round((total_casualties / total_events * 10) if total_events > 0 else 0, 1)
        st.metric("Severity Index", f"{severity_score}/100", help="Casualties per event × 10")
    
    with col3:
        
        if summary.get("active_months"):
            month_counts = Counter(summary["active_months"])
            peak_month_code = max(month_counts.items(), key=lambda x: x[1])[0]
            peak_month_name = calendar.month_name[int(peak_month_code)]
            st.write("### Peak Activity")
            st.write(f"**{peak_month_name}**")
    
    
    generate_dynamic_insights(summary)

def generate_dynamic_insights(summary):
    """Generate insights based on the data"""
    st.write("## 💡 Key Insights & Recommendations")
    
    insights = []
    recommendations = []
    
    
    triggers = summary.get("triggers", [])
    weather_triggers = [t for t in triggers if any(word in t.lower() for word in ['rain', 'monsoon', 'downpour', 'storm'])]
    if weather_triggers:
        weather_percentage = (len(weather_triggers) / len(triggers)) * 100
        insights.append(f"🌧️ **Weather Dependency**: {weather_percentage:.0f}% of triggers are weather-related")
        recommendations.append("Implement robust rainfall monitoring and early warning systems")
    
    
    if summary.get("active_months"):
        monsoon_months = ["06", "07", "08", "09"]
        active_monsoon = [m for m in summary["active_months"] if m in monsoon_months]
        if len(active_monsoon) >= 3:
            insights.append("📅 **Strong Monsoon Correlation**: High activity during June-September period")
            recommendations.append("Enhanced preparedness and monitoring during monsoon season")
    
    
    total_fatalities = summary.get("total_fatalities", 0)
    total_injuries = summary.get("total_injuries", 0)
    if total_fatalities > 0 and total_injuries > 0:
        fatality_ratio = total_fatalities / (total_fatalities + total_injuries)
        if fatality_ratio > 0.8:
            insights.append("⚠️ **High Lethality**: Fatality rate significantly exceeds injury rate")
            recommendations.append("Focus on evacuation protocols and early warning systems for high-impact events")
    
    
    total_events = summary.get("total_events", 0)
    if total_events > 100:
        insights.append("📈 **High Frequency Events**: Significant number of recorded incidents")
        recommendations.append("Comprehensive hazard mapping and risk zoning required")
    
    
    col1, col2 = st.columns(2)
    
    with col1:
        if insights:
            st.write("**🔍 Data Insights:**")
            for insight in insights:
                st.info(insight)
    
    with col2:
        if recommendations:
            st.write("**📋 Recommendations:**")
            for rec in recommendations:
                st.success(f"• {rec}")
    
    if not insights and not recommendations:
        st.info("📊 More comprehensive data needed for detailed insights and recommendations")


def add_custom_css():
    st.markdown("""
    <style>
    .metric-card {
        background-color: 
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid 
        margin-bottom: 1rem;
    }
    .risk-critical { border-left-color: 
    .risk-high { border-left-color: 
    .risk-medium { border-left-color: 
    .risk-low { border-left-color: 
    
    .stExpander > div:first-child {
        background-color: 
        border-radius: 0.5rem;
    }
    
    .insight-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


add_custom_css()