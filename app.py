import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Movie Recommendation System", page_icon="🎬", layout="wide"
)


# 1. Load dataset from GitHub repository
@st.cache_data
def load_data():
  return pd.read_csv("movielens_merged_2.csv")


try:
  df = load_data()

  # Extract all available genres
  genres_set = set()
  if "genres" in df.columns:
    for g in df["genres"].dropna():
      genres_set.update(g.split("|"))
  all_genres = sorted(list(genres_set))

  # Application Header
  st.title("🎬 Movie Recommendation System")
  st.write(
      "Interactive UI for predicting ratings and recommending movies based on"
      " your preferences."
  )

  st.divider()

  # UI Layout Split
  col1, col2 = st.columns([1, 2])

  with col1:
    st.subheader("⚙️ Filter Options")

    # User Selection
    selected_user = st.number_input(
        "Enter User ID:",
        min_value=int(df["userId"].min()),
        max_value=int(df["userId"].max()),
        value=int(df["userId"].min()),
    )

    # Genre Dropdown Selection
    selected_genres = st.multiselect(
        "Filter by Genre:",
        options=all_genres,
        placeholder="Select genres...",
    )

    # Number of Recommendations Slider
    num_recommendations = st.slider(
        "Number of Recommendations:", min_value=1, max_value=20, value=5
    )

    btn_predict = st.button("🚀 Get Recommendations", type="primary")

  # 2. Collaborative Filtering Recommendation Algorithm
  def recommend_movies(user_id, df, selected_genres, k=5, top_n=5):
    filtered_df = df.copy()

    # Filter by genre if selected
    if selected_genres:
      pattern = "|".join(selected_genres)
      filtered_df = filtered_df[
          filtered_df["genres"].str.contains(pattern, na=False)
      ]

    user_item_matrix = filtered_df.pivot_table(
        index="userId", columns="title", values="rating"
    )

    if user_id not in user_item_matrix.index:
      return (
          None,
          "This user has no prior ratings in the selected genre(s)!",
      )

    # Mean-centering & Cosine Similarity calculation
    user_means = user_item_matrix.mean(axis=1)
    matrix_centered = user_item_matrix.sub(user_means, axis=0).fillna(0)

    matrix_bytes = matrix_centered.values
    norms = np.linalg.norm(matrix_bytes, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    normalized_matrix = matrix_bytes / norms
    similarity_matrix = np.dot(normalized_matrix, normalized_matrix.T)

    sim_df = pd.DataFrame(
        similarity_matrix,
        index=user_item_matrix.index,
        columns=user_item_matrix.index,
    )

    user_ratings = user_item_matrix.loc[user_id]
    unrated_movies = user_ratings[user_ratings.isna()].index

    sim_scores = sim_df[user_id].drop(user_id)
    top_k_neighbors = sim_scores.nlargest(k).index

    predicted_ratings = {}
    u_mean = user_means[user_id]

    for movie in unrated_movies:
      neighbor_ratings = user_item_matrix.loc[top_k_neighbors, movie]
      valid_neighbors = neighbor_ratings.dropna()

      if valid_neighbors.empty:
        continue

      weights = sim_scores.loc[valid_neighbors.index]
      sim_sum = weights.abs().sum()

      if sim_sum == 0:
        continue

      centered_ratings = valid_neighbors - user_means.loc[valid_neighbors.index]
      predicted_rating = u_mean + (np.dot(weights, centered_ratings) / sim_sum)

      # Scale bounds between 0.5 and 5.0
      predicted_ratings[movie] = round(
          min(5.0, max(0.5, predicted_rating)), 2
      )

    rec_df = (
        pd.DataFrame(
            list(predicted_ratings.items()),
            columns=["Movie Title", "Predicted Rating"],
        )
        .sort_values(by="Predicted Rating", ascending=False)
        .head(top_n)
    )

    return rec_df, None

  # Results Column
  with col2:
    st.subheader("🎯 Predictions & Recommended Movies")

    if btn_predict:
      with st.spinner("Calculating recommendations..."):
        results, error = recommend_movies(
            selected_user,
            df,
            selected_genres,
            k=5,
            top_n=num_recommendations,
        )

        if error:
          st.error(error)
        elif results.empty:
          st.warning(
              "No sufficient recommendations found. Try adjusting genre filters."
          )
        else:
          st.success("Recommendations generated successfully!")
          st.dataframe(results, use_container_width=True)

except Exception as e:
  st.error(f"Error loading dataset: {e}")
