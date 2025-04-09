import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# 🔑 RAWG API Key
API_KEY = st.secrets["RAWG_API_KEY"]
BASE_URL = "https://api.rawg.io/api"

# Function to fetch games based on filters
def get_games(filters):
    url = f"{BASE_URL}/games?key={API_KEY}&ordering=-rating&page_size=50"
    
    if "genres" in filters and filters["genres"]:
        url += f"&genres={','.join(filters['genres'])}"
    if "platform" in filters and filters["platform"]:
        url += f"&platforms={filters['platform']}"
    if "release_date" in filters and filters["release_date"]:
        url += f"&dates={filters['release_date']}"
    
    response = requests.get(url).json()
    return response.get("results", [])

# Function to fetch platforms
def get_platforms():
    url = f"{BASE_URL}/platforms?key={API_KEY}"
    response = requests.get(url).json()
    return response.get("results", [])

# Function to fetch all genres
def get_genres():
    url = f"{BASE_URL}/genres?key={API_KEY}"
    response = requests.get(url).json()
    return response.get("results", [])

# Function to fetch all game companies (both publishers and developers)
def get_all_game_companies():
    url_publishers = f"{BASE_URL}/publishers?key={API_KEY}"
    url_developers = f"{BASE_URL}/developers?key={API_KEY}"
    
    publishers = requests.get(url_publishers).json().get("results", [])
    developers = requests.get(url_developers).json().get("results", [])

    all_companies = {c["name"] for c in publishers + developers}  # Using set to avoid duplicates
    return sorted(list(all_companies))

# Streamlit App
st.set_page_config(page_title="Advanced Game Analytics Dashboard", layout="wide")

st.title("🎮 Advanced Game Analytics Dashboard")

# Sidebar Navigation
menu = st.sidebar.radio("Navigation", [
    "📊 Game Popularity & Trends",
    "📊 Genre Distribution Analysis",
    "📈 Game Popularity Prediction",
    "📈 Game Lifecycle Analysis"
])

# Section 1: Game Popularity & Trends
if menu == "📊 Game Popularity & Trends":
    st.header("🔥 Explore Popular Games with Advanced Filters")

    # Fetch available genres and platforms
    genres = get_genres()
    genre_options = ["All Genres"] + [g["slug"] for g in genres]  
    selected_genres = st.multiselect("🎭 Select Genres", genre_options, default="All Genres")

    if "All Genres" in selected_genres:
        selected_genres = []  

    platforms = get_platforms()
    platform_options = ["All Platforms"] + [p["slug"] for p in platforms]  
    selected_platform = st.selectbox("🕹️ Select Platform", platform_options, index=0)

    if selected_platform == "All Platforms":
        selected_platform = None  

    today = datetime.today().date()
    one_year_ago = (datetime.today() - timedelta(days=365)).date()
    
    release_date = st.slider(
        "📅 Select Release Date Range",
        min_value=datetime(2000, 1, 1).date(),  
        max_value=today,
        value=(one_year_ago, today),
        format="YYYY-MM-DD"
    )

    start_date_str = release_date[0].strftime("%Y-%m-%d")
    end_date_str = release_date[1].strftime("%Y-%m-%d")

    min_rating, max_rating = st.slider(
        "⭐ Select Rating Range",
        min_value=0.0, max_value=5.0,
        value=(0.0, 5.0), step=0.1
    )

    filters = {
        "genres": selected_genres,
        "platform": selected_platform,
        "release_date": f"{start_date_str},{end_date_str}"
    }
    filtered_games = get_games(filters)

    filtered_games = [game for game in filtered_games if min_rating <= game.get("rating", 0) <= max_rating]

    if filtered_games:
        st.subheader("📌 Filtered Games")
        df_filtered = pd.DataFrame(filtered_games)[["name", "rating", "released"]]
        st.dataframe(df_filtered)

        fig = px.bar(df_filtered, x="name", y="rating", title="📊 Game Ratings", color="rating", height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No games found with selected filters.")

# Section 2: Genre Distribution Analysis
elif menu == "📊 Genre Distribution Analysis":
    st.header("📊 Analyze Game Genres Popularity")

    genres_data = get_genres()

    if genres_data:
        genre_names = [g["name"] for g in genres_data]
        game_counts = [g["games_count"] for g in genres_data]

        df_genres = pd.DataFrame({"Genre": genre_names, "Game Count": game_counts})

        fig_pie = px.pie(df_genres, names="Genre", values="Game Count", title="🎭 Genre Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)

        fig_bar = px.bar(df_genres, x="Genre", y="Game Count", title="📊 Popular Game Genres",
                         color="Game Count", height=500)
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning("No genre data available.")

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.header("📈 Game Popularity Prediction System")

# Fetch all game companies
companies = get_all_game_companies()
company_options = ["Select a Company"] + companies
selected_company = st.selectbox("🏢 Select Game Company", company_options, index=0)

# User inputs for feature values
platform_users = st.number_input("👥 Estimated Platform Users (in millions)", min_value=0, step=1, value=80)
critic_reviews = st.number_input("📝 Number of Critic Reviews", min_value=0, step=1, value=500)
genre_popularity = st.slider("🎭 Genre Popularity (1-10)", min_value=1, max_value=10, value=8)
average_rating = st.slider("⭐ Average Rating (0-5)", min_value=0.0, max_value=5.0, value=4.2, step=0.1)
budget = st.number_input("💰 Development Budget (in million USD)", min_value=1, step=1, value=100)
playtime = st.number_input("⏳ Average Playtime (in hours)", min_value=1, step=1, value=50)
metacritic_score = st.slider("🎯 Metacritic Score (0-100)", min_value=0, max_value=100, value=85, step=1)

# Predict Popularity
if st.button("🔮 Predict Game Popularity"):
    # Generate a larger dummy dataset
    np.random.seed(42)
    data_size = 500  # Increased dataset size for better training
    df_dummy = pd.DataFrame({
        "company": np.random.choice(company_options[1:], data_size),
        "platform_users": np.random.randint(10, 500, data_size),
        "genre_popularity": np.random.randint(1, 10, data_size),
        "critic_reviews": np.random.randint(50, 5000, data_size),
        "average_rating": np.random.uniform(0, 5, data_size),
        "budget": np.random.randint(1, 500, data_size),
        "playtime": np.random.randint(1, 200, data_size),
        "metacritic_score": np.random.randint(0, 100, data_size),
        "popularity_score": np.random.randint(10, 100, data_size)  # More diverse score distribution
    })

    # Encode categorical values
    encoder = LabelEncoder()
    df_dummy["company"] = encoder.fit_transform(df_dummy["company"])

    # Scale numerical features
    scaler = StandardScaler()
    feature_cols = ["platform_users", "genre_popularity", "critic_reviews", "average_rating", 
                    "budget", "playtime", "metacritic_score"]
    df_dummy[feature_cols] = scaler.fit_transform(df_dummy[feature_cols])

    # Train Model
    X = df_dummy.drop("popularity_score", axis=1)
    y = df_dummy["popularity_score"]
    model = RandomForestRegressor(n_estimators=100, random_state=42)  # More accurate model
    model.fit(X, y)

    # Predict
    if selected_company != "Select a Company":
        user_data = np.array([[encoder.transform([selected_company])[0], platform_users, genre_popularity, 
                               critic_reviews, average_rating, budget, playtime, metacritic_score]])

        user_data[:, 1:] = scaler.transform(user_data[:, 1:])  # Scale user input

        predicted_popularity = model.predict(user_data)[0]
        
        # Display Feature Importance
        feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

        st.success(f"🔥 Predicted Game Popularity Score: {round(predicted_popularity, 2)} / 100")
        
        # Show Feature Importance
        st.subheader("🔍 Feature Importance")
        st.bar_chart(feature_importance)

    else:
        st.warning("⚠️ Please select a game company before predicting.")

# Section 4: Game Lifecycle Analysis
if menu == "📈 Game Lifecycle Analysis":
    st.header("📈 Game Lifecycle & Popularity Trends")

    # User input for number of years to analyze
    years_range = st.slider("📅 Select Year Range for Analysis", min_value=2000, max_value=2025, value=(2010, 2025))

    # Fetch popular games based on release year
    games_data = get_games({"release_date": f"{years_range[0]}-01-01,{years_range[1]}-12-31"})

    if games_data:
        df_games = pd.DataFrame(games_data)

        # Convert release date to datetime
        df_games["released"] = pd.to_datetime(df_games["released"])
        df_games["year"] = df_games["released"].dt.year

        # Group data to analyze popularity trends
        popularity_trend = df_games.groupby("year").agg(
            avg_rating=("rating", "mean"),
            total_reviews=("ratings_count", "sum"),
            game_count=("id", "count")
        ).reset_index()

        # Line chart for Average Ratings Over Time
        fig_ratings = px.line(
            popularity_trend, x="year", y="avg_rating",
            title="⭐ Average Game Ratings Over Time",
            markers=True
        )
        st.plotly_chart(fig_ratings, use_container_width=True)

        # Line chart for Total Reviews Over Time (Popularity)
        fig_reviews = px.line(
            popularity_trend, x="year", y="total_reviews",
            title="💬 Total User Reviews Over Time (Popularity Indicator)",
            markers=True
        )
        st.plotly_chart(fig_reviews, use_container_width=True)

        # Bar chart for Number of Games Released Each Year
        fig_game_count = px.bar(
            popularity_trend, x="year", y="game_count",
            title="🎮 Number of Games Released Per Year",
            color="game_count",
            height=500
        )
        st.plotly_chart(fig_game_count, use_container_width=True)

    else:
        st.warning("No game data available for the selected years.")
