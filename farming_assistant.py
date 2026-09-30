import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from datetime import datetime
import json

class FarmingAssistant:
    def __init__(self):
        self.crop_data = self.load_crop_data()
        self.soil_types = ['Sandy', 'Loamy', 'Clay', 'Silty', 'Peaty', 'Chalky']
        self.climate_zones = ['Tropical', 'Subtropical', 'Temperate', 'Arid', 'Mediterranean']
        self.setup_ml_models()
    
    def load_crop_data(self):
        """Load crop database with recommendations"""
        data = {
            'crop': ['Rice', 'Wheat', 'Maize', 'Cotton', 'Sugarcane', 'Soybean', 
                    'Potato', 'Tomato', 'Onion', 'Carrot', 'Cabbage', 'Spinach',
                    'Apple', 'Banana', 'Mango', 'Orange', 'Grapes', 'Strawberry'],
            'soil_type': ['Clay', 'Loamy', 'Loamy', 'Sandy', 'Loamy', 'Loamy',
                         'Sandy', 'Loamy', 'Sandy', 'Sandy', 'Loamy', 'Loamy',
                         'Loamy', 'Sandy', 'Loamy', 'Loamy', 'Sandy', 'Loamy'],
            'climate': ['Tropical', 'Temperate', 'Tropical', 'Arid', 'Tropical', 'Temperate',
                       'Temperate', 'Subtropical', 'Temperate', 'Temperate', 'Temperate', 'Temperate',
                       'Temperate', 'Tropical', 'Tropical', 'Subtropical', 'Mediterranean', 'Temperate'],
            'water_requirement': ['High', 'Medium', 'Medium', 'Low', 'High', 'Medium',
                                 'Medium', 'Medium', 'Low', 'Medium', 'High', 'High',
                                 'Medium', 'High', 'Medium', 'Medium', 'Low', 'High'],
            'season': ['Kharif', 'Rabi', 'Kharif', 'Kharif', 'YearRound', 'Kharif',
                      'Rabi', 'YearRound', 'Rabi', 'Rabi', 'YearRound', 'Winter',
                      'YearRound', 'YearRound', 'Summer', 'Winter', 'Summer', 'Spring'],
            'yield_potential': ['High', 'High', 'Very High', 'Medium', 'High', 'Medium',
                               'High', 'Very High', 'Medium', 'Medium', 'High', 'Medium',
                               'Medium', 'High', 'High', 'Medium', 'Medium', 'High']
        }
        return pd.DataFrame(data)
    
    def setup_ml_models(self):
        """Setup machine learning models for crop prediction"""
        # Prepare data for ML model
        df = self.crop_data.copy()
        le_soil = LabelEncoder()
        le_climate = LabelEncoder()
        le_water = LabelEncoder()
        
        df['soil_encoded'] = le_soil.fit_transform(df['soil_type'])
        df['climate_encoded'] = le_climate.fit_transform(df['climate'])
        df['water_encoded'] = le_water.fit_transform(df['water_requirement'])
        
        X = df[['soil_encoded', 'climate_encoded', 'water_encoded']]
        y = df['crop']
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        
        self.le_soil = le_soil
        self.le_climate = le_climate
        self.le_water = le_water
    
    def predict_best_crops(self, soil_type, climate_zone, water_availability):
        """Predict best crops based on conditions"""
        try:
            soil_encoded = self.le_soil.transform([soil_type])[0]
            climate_encoded = self.le_climate.transform([climate_zone])[0]
            water_encoded = self.le_water.transform([water_availability])[0]
            
            features = np.array([[soil_encoded, climate_encoded, water_encoded]])
            probabilities = self.model.predict_proba(features)[0]
            
            crop_probs = list(zip(self.model.classes_, probabilities))
            crop_probs.sort(key=lambda x: x[1], reverse=True)
            
            return crop_probs[:5]  # Top 5 recommendations
        except:
            # Fallback to rule-based matching
            filtered_crops = self.crop_data[
                (self.crop_data['soil_type'] == soil_type) & 
                (self.crop_data['climate'] == climate_zone) &
                (self.crop_data['water_requirement'] == water_availability)
            ]
            return [(crop, 0.8) for crop in filtered_crops['crop'].head(5)]
    
    def get_crop_recommendations(self, crop_name):
        """Get detailed recommendations for specific crop"""
        crop_info = self.crop_data[self.crop_data['crop'] == crop_name].iloc[0]
        
        recommendations = {
            'Soil Preparation': f"Prepare {crop_info['soil_type']} soil with proper drainage",
            'Sowing Time': f"Best sown in {crop_info['season']} season",
            'Water Management': f"{crop_info['water_requirement']} water requirements. Implement drip irrigation for efficiency",
            'Fertilizer': "Use balanced NPK fertilizer with organic compost",
            'Pest Control': "Regular monitoring and integrated pest management",
            'Harvesting': f"Expected yield: {crop_info['yield_potential']}. Harvest when mature",
            'Best Practices': [
                "Soil testing before planting",
                "Crop rotation to maintain soil health",
                "Use of certified seeds",
                "Regular weeding and maintenance"
            ]
        }
        
        # Add crop-specific tips
        specific_tips = self.get_crop_specific_tips(crop_name)
        recommendations.update(specific_tips)
        
        return recommendations
    
    def get_crop_specific_tips(self, crop_name):
        """Get specific tips for different crops"""
        tips = {
            'Rice': {
                'Special Notes': 'Requires standing water, ideal pH 5-6.5',
                'Key Practices': ['Transplanting', 'Water logging management', 'Balanced nitrogen application']
            },
            'Wheat': {
                'Special Notes': 'Cool season crop, requires well-drained soil',
                'Key Practices': ['Timely sowing', 'Weed control', 'Proper irrigation scheduling']
            },
            'Tomato': {
                'Special Notes': 'Warm season crop, requires staking',
                'Key Practices': ['Regular pruning', 'Disease monitoring', 'Consistent watering']
            },
            'Maize': {
                'Special Notes': 'Warm weather crop, requires deep soil',
                'Key Practices': ['Proper spacing', 'Nitrogen management', 'Weed control']
            },
            'Cotton': {
                'Special Notes': 'Long duration crop, requires warm climate',
                'Key Practices': ['Pest management', 'Proper irrigation', 'Timely harvesting']
            }
        }
        return tips.get(crop_name, {
            'Special Notes': 'Follow general best practices for optimal yield',
            'Key Practices': ['Soil testing', 'Proper spacing', 'Regular monitoring']
        })
    
    def calculate_irrigation_schedule(self, crop_name, soil_type, climate):
        """Calculate optimal irrigation schedule"""
        base_schedule = {
            'Rice': {'frequency': 'Daily', 'duration': '2-3 hours', 'method': 'Flood irrigation'},
            'Wheat': {'frequency': 'Every 5-7 days', 'duration': '1-2 hours', 'method': 'Sprinkler'},
            'Tomato': {'frequency': 'Every 3-4 days', 'duration': '30-45 minutes', 'method': 'Drip'},
            'Maize': {'frequency': 'Every 4-6 days', 'duration': '1-1.5 hours', 'method': 'Sprinkler'},
            'Cotton': {'frequency': 'Every 7-10 days', 'duration': '1 hour', 'method': 'Drip/Sprinkler'},
            'Sugarcane': {'frequency': 'Every 5-7 days', 'duration': '2-3 hours', 'method': 'Flood'},
            'Potato': {'frequency': 'Every 4-5 days', 'duration': '45-60 minutes', 'method': 'Drip'}
        }
        
        schedule = base_schedule.get(crop_name, {
            'frequency': 'Weekly', 
            'duration': '1 hour', 
            'method': 'Drip irrigation'
        })
        
        # Adjust based on soil and climate
        if soil_type == 'Sandy':
            schedule['frequency'] = 'More frequent, shorter duration'
        if climate == 'Arid':
            schedule['duration'] = 'Increase by 20%'
            
        return schedule
    
    def get_weather_insights(self):
        """Get weather insights for farming decisions"""
        return {
            'General Advice': [
                "Monitor weather forecasts regularly",
                "Plan irrigation based on rainfall predictions",
                "Protect crops from extreme weather events",
                "Use weather-based pest and disease warnings"
            ],
            'Seasonal Tips': {
                'Summer': 'Increase irrigation frequency, mulch soil, protect from heat stress',
                'Winter': 'Protect from frost, reduce irrigation, use row covers',
                'Monsoon': 'Ensure drainage, prevent waterlogging, manage fungal diseases',
                'Spring': 'Ideal for most crops, focus on timely planting'
            }
        }

def main():
    st.set_page_config(
        page_title="AI Farming Assistant",
        page_icon="🌱",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🌱 AI Farming Assistant")
    st.markdown("### Your Intelligent Partner for Modern Agriculture")
    
    # Initialize assistant
    assistant = FarmingAssistant()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose Service", 
        ["Crop Recommendation", "Crop Management", "Irrigation Planning", "Weather Insights", "Market Analysis"])
    
    if app_mode == "Crop Recommendation":
        show_crop_recommendation(assistant)
    elif app_mode == "Crop Management":
        show_crop_management(assistant)
    elif app_mode == "Irrigation Planning":
        show_irrigation_planning(assistant)
    elif app_mode == "Weather Insights":
        show_weather_insights(assistant)
    elif app_mode == "Market Analysis":
        show_market_analysis()

def show_crop_recommendation(assistant):
    st.header("🌾 Smart Crop Recommendation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Farm Conditions")
        soil_type = st.selectbox("Soil Type", assistant.soil_types)
        climate_zone = st.selectbox("Climate Zone", assistant.climate_zones)
        water_availability = st.selectbox("Water Availability", ["Low", "Medium", "High"])
        farm_size = st.number_input("Farm Size (acres)", min_value=0.1, value=5.0, step=0.1)
        
    with col2:
        st.subheader("Additional Factors")
        budget = st.selectbox("Budget Level", ["Low", "Medium", "High"])
        experience = st.selectbox("Farming Experience", ["Beginner", "Intermediate", "Expert"])
        market_demand = st.selectbox("Market Demand Priority", ["Low", "Medium", "High"])
    
    if st.button("Get Crop Recommendations", type="primary"):
        with st.spinner("Analyzing optimal crops for your farm..."):
            recommendations = assistant.predict_best_crops(soil_type, climate_zone, water_availability)
            
            st.success("🎯 Top Crop Recommendations for Your Farm")
            
            for i, (crop, probability) in enumerate(recommendations, 1):
                with st.expander(f"{i}. {crop} (Suitability: {probability:.1%})", expanded=i==1):
                    crop_info = assistant.get_crop_recommendations(crop)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Key Requirements:**")
                        st.write(f"• Soil: {assistant.crop_data[assistant.crop_data['crop'] == crop]['soil_type'].iloc[0]}")
                        st.write(f"• Climate: {assistant.crop_data[assistant.crop_data['crop'] == crop]['climate'].iloc[0]}")
                        st.write(f"• Water: {assistant.crop_data[assistant.crop_data['crop'] == crop]['water_requirement'].iloc[0]}")
                        st.write(f"• Season: {assistant.crop_data[assistant.crop_data['crop'] == crop]['season'].iloc[0]}")
                    
                    with col2:
                        st.write("**Management Practices:**")
                        for practice in crop_info['Best Practices']:
                            st.write(f"• {practice}")
                    
                    st.write("**Detailed Guidelines:**")
                    for key, value in crop_info.items():
                        if key not in ['Best Practices', 'Special Notes', 'Key Practices']:
                            if isinstance(value, list):
                                for item in value:
                                    st.write(f"• **{key}:** {item}")
                            else:
                                st.write(f"• **{key}:** {value}")
                    
                    # Show special notes if available
                    if 'Special Notes' in crop_info:
                        st.info(f"**Special Notes:** {crop_info['Special Notes']}")
                    
                    # Show key practices if available
                    if 'Key Practices' in crop_info:
                        st.write("**Key Practices:**")
                        for practice in crop_info['Key Practices']:
                            st.write(f"• {practice}")

def show_crop_management(assistant):
    st.header("📊 Crop Management Guide")
    
    selected_crop = st.selectbox("Select Your Crop", assistant.crop_data['crop'].unique())
    
    if selected_crop:
        st.success(f"Management Guide for {selected_crop}")
        
        recommendations = assistant.get_crop_recommendations(selected_crop)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌱 Growth Requirements")
            for key in ['Soil Preparation', 'Sowing Time', 'Water Management']:
                st.info(f"**{key}:** {recommendations[key]}")
        
        with col2:
            st.subheader("⚡ Management Practices")
            for key in ['Fertilizer', 'Pest Control', 'Harvesting']:
                st.info(f"**{key}:** {recommendations[key]}")
        
        st.subheader("📋 Best Practices")
        practices_col1, practices_col2 = st.columns(2)
        
        with practices_col1:
            for practice in recommendations['Best Practices'][:len(recommendations['Best Practices'])//2]:
                st.write(f"✅ {practice}")
        
        with practices_col2:
            for practice in recommendations['Best Practices'][len(recommendations['Best Practices'])//2:]:
                st.write(f"✅ {practice}")
        
        # Special notes and key practices
        if 'Special Notes' in recommendations:
            st.subheader("💡 Special Notes")
            st.warning(recommendations['Special Notes'])
        
        if 'Key Practices' in recommendations:
            st.subheader("🔑 Key Practices")
            for practice in recommendations['Key Practices']:
                st.write(f"🎯 {practice}")
        
        # Growth timeline
        st.subheader("📅 Growth Timeline")
        timeline_data = {
            'Stage': ['Preparation', 'Sowing', 'Vegetative', 'Flowering', 'Fruiting', 'Harvesting'],
            'Duration': ['1-2 weeks', '1 week', '3-8 weeks', '1-2 weeks', '2-6 weeks', '1-2 weeks'],
            'Tasks': [
                'Soil testing, plowing, fertilization',
                'Seed treatment, sowing',
                'Weeding, irrigation, pest control',
                'Pollination management',
                'Fruit development monitoring',
                'Harvest, post-harvest handling'
            ]
        }
        
        timeline_df = pd.DataFrame(timeline_data)
        st.dataframe(timeline_df, use_container_width=True)

def show_irrigation_planning(assistant):
    st.header("💧 Smart Irrigation Planning")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_crop = st.selectbox("Select Crop", assistant.crop_data['crop'].unique(), key="irrigation_crop")
        soil_type = st.selectbox("Soil Type", assistant.soil_types, key="irrigation_soil")
        climate = st.selectbox("Climate", assistant.climate_zones, key="irrigation_climate")
        area = st.number_input("Area to irrigate (acres)", min_value=0.1, value=1.0, step=0.1)
    
    with col2:
        water_source = st.selectbox("Water Source", ["Well", "Canal", "River", "Rainwater", "Borewell"])
        irrigation_type = st.selectbox("Irrigation System", ["Drip", "Sprinkler", "Flood", "Manual"])
        rainfall = st.slider("Expected Rainfall (mm/month)", 0, 500, 100)
    
    if st.button("Generate Irrigation Plan"):
        schedule = assistant.calculate_irrigation_schedule(selected_crop, soil_type, climate)
        
        st.success("💧 Optimal Irrigation Schedule")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Frequency", schedule['frequency'])
        with col2:
            st.metric("Duration", schedule['duration'])
        with col3:
            st.metric("Method", schedule['method'])
        
        # Water requirement calculation
        water_requirements = {
            'Rice': 1200, 'Wheat': 450, 'Maize': 500, 'Cotton': 600,
            'Sugarcane': 1500, 'Tomato': 400, 'Potato': 350, 'Soybean': 450,
            'Onion': 350, 'Carrot': 400, 'Cabbage': 500, 'Spinach': 300
        }
        
        base_water = water_requirements.get(selected_crop, 500)
        adjusted_water = base_water - (rainfall * 0.8)  # Adjust for rainfall
        
        st.info(f"**Estimated water requirement:** {max(adjusted_water, 100):.0f} mm/season")
        
        # Water savings calculation
        if irrigation_type == "Drip":
            savings = adjusted_water * 0.3
            st.success(f"💧 **Drip irrigation can save:** ~{savings:.0f} mm/season (30% savings)")
        
        # Efficiency tips
        st.subheader("💡 Water Efficiency Tips")
        tips = [
            "Use mulching to reduce evaporation by 25-50%",
            "Implement drip irrigation for water-intensive crops",
            "Schedule irrigation during early morning or late evening",
            "Monitor soil moisture regularly with sensors",
            "Use rainwater harvesting systems",
            "Practice deficit irrigation during non-critical growth stages"
        ]
        
        for tip in tips:
            st.write(f"• {tip}")

def show_weather_insights(assistant):
    st.header("🌤️ Weather & Climate Insights")
    
    insights = assistant.get_weather_insights()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 General Weather Advice")
        for advice in insights['General Advice']:
            st.write(f"• {advice}")
    
    with col2:
        st.subheader("🌦️ Seasonal Recommendations")
        for season, tips in insights['Seasonal Tips'].items():
            with st.expander(f"{season} Season"):
                st.write(tips)
    
    # Soil moisture monitoring
    st.subheader("🌱 Soil Moisture Management")
    
    soil_moisture_data = {
        'Soil Type': ['Sandy', 'Loamy', 'Clay', 'Silty'],
        'Field Capacity (%)': [15, 35, 45, 40],
        'Wilting Point (%)': [5, 15, 25, 20],
        'Ideal Moisture (%)': [10-12, 25-30, 30-35, 25-30]
    }
    
    moisture_df = pd.DataFrame(soil_moisture_data)
    st.dataframe(moisture_df, use_container_width=True)
    
    # Weather alerts simulation
    st.subheader("🚨 Current Weather Alerts")
    
    alerts = [
        {"type": "⚠️", "message": "Moderate rainfall expected in next 48 hours", "action": "Delay irrigation and prepare drainage"},
        {"type": "🌡️", "message": "Temperature rising above seasonal average", "action": "Increase crop monitoring for heat stress"},
        {"type": "💨", "message": "Moderate winds predicted this week", "action": "Secure young plants and check supports"}
    ]
    
    for alert in alerts:
        st.warning(f"{alert['type']} **Alert:** {alert['message']} \n\n**Recommended Action:** {alert['action']}")

def show_market_analysis():
    st.header("📈 Market Analysis & Pricing")
    
    # Simulated market data
    crops_market_data = {
        'Crop': ['Wheat', 'Rice', 'Tomato', 'Potato', 'Onion', 'Cotton', 'Maize', 'Soybean'],
        'Current Price (₹/kg)': [25, 35, 20, 15, 30, 65, 18, 40],
        'Demand Trend': ['Stable', 'High', 'Seasonal', 'Stable', 'Volatile', 'Growing', 'Stable', 'Increasing'],
        'Best Market': ['Mandi', 'Wholesale', 'Local', 'Mandi', 'Local', 'Export', 'Mandi', 'Processing'],
        'Storage Advice': ['6-12 months', '12-18 months', '1-2 weeks', '3-6 months', '2-3 months', 'Long-term', '6-9 months', '9-12 months']
    }
    
    market_df = pd.DataFrame(crops_market_data)
    st.dataframe(market_df, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💡 Marketing Strategies")
        strategies = [
            "Direct to consumer through farmers markets",
            "Contract farming with agribusiness companies",
            "Online platforms for produce sales",
            "Value addition through processing (jams, pickles, flour)",
            "Cooperative marketing for better prices",
            "Export opportunities for quality produce"
        ]
        
        for strategy in strategies:
            st.write(f"• {strategy}")
    
    with col2:
        st.subheader("📊 Price Forecasting Tips")
        st.info("""
        **Short-term (1-3 months):**
        - Seasonal vegetables: Prices may fluctuate based on supply
        - Grains: Stable to slightly increasing
        - Fruits: Monitor seasonal availability
        
        **Long-term (6-12 months):**
        - Monitor government agricultural policies
        - Track export opportunities and international prices
        - Consider climate impact on production
        - Watch input cost trends (fertilizers, fuel)
        """)
    
    st.subheader("🛒 Value Addition Opportunities")
    value_additions = {
        'Fruits': ['Juices', 'Jams', 'Dried fruits', 'Pickles'],
        'Vegetables': ['Frozen products', 'Pickles', 'Dehydrated snacks', 'Sauces'],
        'Grains': ['Flour', 'Ready-to-cook products', 'Breakfast cereals', 'Bakery products'],
        'Others': ['Organic certification', 'Farm tourism', 'Seed production', 'Herbal products']
    }
    
    for category, opportunities in value_additions.items():
        with st.expander(f"Value Addition for {category}"):
            for opportunity in opportunities:
                st.write(f"• {opportunity}")

# Additional utility functions
def create_dashboard():
    """Create a comprehensive farming dashboard"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Farm Statistics")
    
    # Simulated farm data
    total_area = st.sidebar.number_input("Total Farm Area (acres)", 1, 1000, 50)
    crops_grown = st.sidebar.multiselect("Current Crops", 
                                        ['Wheat', 'Rice', 'Vegetables', 'Fruits', 'Pulses', 'Cotton', 'Sugarcane'])
    
    if crops_grown:
        st.sidebar.success(f"Crops: {', '.join(crops_grown)}")
    
    st.sidebar.metric("Estimated Yield", f"{(total_area * 2.5):.1f} tons")
    st.sidebar.metric("Water Requirement", f"{(total_area * 1000):,} liters/day")
    
    # Quick tips in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("💡 Quick Tips")
    st.sidebar.info("""
    - Test soil before planting
    - Use quality seeds
    - Monitor crops regularly
    - Keep weather updates handy
    """)

if __name__ == "__main__":
    main()
    create_dashboard()
