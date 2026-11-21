
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Movie Intelligence",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Personalized Movie Intelligence System")
st.write("Discover movies based on your interests!")

@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    return movies, ratings

movies, ratings = load_data()

movies["genres"] = movies["genres"].fillna("")
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres"])

cosine_sim = cosine_similarity(tfidf_matrix)

st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Select Page",
    ["Home", "Movie Recommendation", "Popular Movies", "Analytics"]
)

if page == "Home":
    st.header("Welcome to Movie Intelligence")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Movies", len(movies))
    col2.metric("Total Ratings", len(ratings))
    col3.metric("Total Users", ratings["userId"].nunique())

    st.write("This project uses Machine Learning to recommend movies.")

elif page == "Movie Recommendation":
    st.header("Find Similar Movies")

    movie_name = st.selectbox(
        "Select a Movie",
        movies["title"].sort_values().tolist()
    )

    if st.button("Recommend Movies"):
        idx = movies[movies["title"] == movie_name].index[0]

        scores = list(enumerate(cosine_sim[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        indices = [i[0] for i in scores[1:11]]

        st.subheader("Recommended Movies")
        st.dataframe(
            movies.iloc[indices][["title", "genres"]],
            use_container_width=True
        )

elif page == "Popular Movies":
    st.header("Top Popular Movies")

    popular = ratings.groupby("movieId").agg(
        average_rating=("rating", "mean"),
        total_ratings=("rating", "count")
    ).reset_index()

    popular = popular[popular["total_ratings"] >= 50]

    popular = popular.merge(movies, on="movieId")
    popular = popular.sort_values(
        "average_rating", ascending=False
    )

    st.dataframe(
        popular[["title", "genres", "average_rating", "total_ratings"]].head(20),
        use_container_width=True
    )

elif page == "Analytics":
    st.header("Movie Rating Analytics")

    st.subheader("Rating Distribution")
    st.bar_chart(ratings["rating"].value_counts().sort_index())

    st.subheader("Top 10 Most Rated Movies")

    top = ratings["movieId"].value_counts().head(10)
    top = top.reset_index()
    top.columns = ["movieId", "total_ratings"]

    top = top.merge(movies, on="movieId")

    st.dataframe(
        top[["title", "total_ratings"]],
        use_container_width=True
    )
