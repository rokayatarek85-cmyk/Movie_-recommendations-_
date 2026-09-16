import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("🎬 Movie Recommendation System")
st.write(
    "نظام توصية الأفلام المعتمد على تشابه المستخدمين (Collaborative Filtering)"
)


# تحميل البيانات أوتوماتيكياً بدون الحاجة لزر رفع
@st.cache_data
def load_data():
  # اكتب اسم ملف الـ CSV المرفوع على GitHub بالضبط بين التنصيص
  return pd.read_csv('movielens_merged.csv')


try:
  df = load_data()
  st.success("✅ تم تحميل البيانات تلقائياً بنجاح!")

  def recommend_movies(user_id, df, k=5, top_n=5):
    user_item_matrix = df.pivot_table(
        index='userId', columns='title', values='rating'
    )

    if user_id not in user_item_matrix.index:
      return None, 'المستخدم غير موجود في البيانات!'

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
      predicted_ratings[movie] = predicted_rating

    rec_df = (
        pd.DataFrame(
            list(predicted_ratings.items()),
            columns=['Title', 'Predicted Rating'],
        )
        .sort_values(by='Predicted Rating', ascending=False)
        .head(top_n)
    )

    return rec_df, None

  col1, col2 = st.columns(2)
  with col1:
    selected_user = st.number_input(
        'أدخل معرف المستخدم (User ID):',
        min_value=int(df['userId'].min()),
        max_value=int(df['userId'].max()),
        value=int(df['userId'].min()),
    )
  with col2:
    num_recommendations = st.slider(
        'عدد التوصيات المطلوبة:', min_value=1, max_value=20, value=5
    )

  if st.button('🔍 عرض التوصيات'):
    with st.spinner('جاري حساب التشابه والتوصيات...'):
      results, error = recommend_movies(
          selected_user, df, k=5, top_n=num_recommendations
      )

      if error:
        st.error(error)
      elif results.empty:
        st.warning('لم يتم العثور على توصيات كافية لهذا المستخدم.')
      else:
        st.subheader(f'🎯 أفضل الأفلام المقترحة للمستخدم {selected_user}:')
        st.dataframe(results, use_container_width=True)

except Exception as e:
  st.error(
      f"تأكدي من رفع ملف الـ CSV على GitHub بنفس الاسم المحدد في الكود ('movielens_merged.csv'). الخطأ: {e}"
  )
