import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Movie Recommendation System", page_icon="🎬", layout="wide"
)


# 1. تحميل البيانات أوتوماتيكياً في الذاكرة لتسريع الأداء
@st.cache_data
def load_data():
  # تأكدي إن اسم الملف مطابق للموجود على GitHub
  return pd.read_csv("movielens_merged.csv")


try:
  df = load_data()

  # استخراج قائمة كل الـ Genres الموجودة في البيانات
  genres_set = set()
  if "genres" in df.columns:
    for g in df["genres"].dropna():
      genres_set.update(g.split("|"))
  all_genres = sorted(list(genres_set))

  # واجهة التطبيق
  st.title("🎬 Movie Recommendation System")
  st.write("واجهة تفاعلية لتوقّع التقييمات واقتراح الأفلام بناءً على تفضيلاتك")

  st.divider()

  # تقسيم الشاشة لأعمدة تفاعلية
  col1, col2 = st.columns([1, 2])

  with col1:
    st.subheader("⚙️ خيارات التصفية")

    # اختيار المستخدم
    selected_user = st.number_input(
        "أدخل معرف المستخدم (User ID):",
        min_value=int(df["userId"].min()),
        max_value=int(df["userId"].max()),
        value=int(df["userId"].min()),
    )

    # قائمة منسدلة (Multiselect Dropdown) للـ Genres
    selected_genres = st.multiselect(
        "تصفية حسب التصنيف (Genre):",
        options=all_genres,
        placeholder="اختر التصنيفات...",
    )

    # عدد التوصيات المطلوبة
    num_recommendations = st.slider(
        "عدد التوصيات المطلوبة:", min_value=1, max_value=20, value=5
    )

    btn_predict = st.button("🚀 عرض التوصيات والتوقعات", type="primary")

  # 2. خوارزمية التوصية والتوقع (Collaborative Filtering)
  def recommend_movies(user_id, df, selected_genres, k=5, top_n=5):
    # تصفية البيانات حسب الـ Genres لو تم اختيارها
    filtered_df = df.copy()
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
          "المستخدم لا يمتلك تقييمات سابقة في هذه التصنيفات المختارة!",
      )

    # حساب المتوسطات والـ Cosine Similarity
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
      # حصر التقييم بين 0.5 و 5
      predicted_ratings[movie] = round(
          min(5.0, max(0.5, predicted_rating)), 2
      )

    rec_df = (
        pd.DataFrame(
            list(predicted_ratings.items()),
            columns=["اسم الفيلم (Movie Title)", "التقييم المتوقع (Predicted Rating)"],
        )
        .sort_values(by="التقييم المتوقع (Predicted Rating)", ascending=False)
        .head(top_n)
    )

    return rec_df, None

  # عرض النتائج في العمود الثاني
  with col2:
    st.subheader("🎯 التوقعات والأفلام المقترحة")

    if btn_predict:
      with st.spinner("جاري حساب التوقعات في ثوانٍ..."):
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
              "لم نجد توصيات كافية تطابق هذه الاختيارات، جربي تقليل التصنيفات."
          )
        else:
          st.success("تم حساب التوقعات بنجاح!")
          st.dataframe(results, use_container_width=True)

except Exception as e:
  st.error(
      f"تأكدي من وجود ملف 'movielens_merged.csv' داخل المستودع على GitHub. التفاصيل: {e}"
  )
