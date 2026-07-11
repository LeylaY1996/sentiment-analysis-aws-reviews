# ============================================================
# ABLATION: Duygu filtreli vs duygu filtresiz öneri sistemi
# (filtered_data zaten notebook'ta tanımlı olmalı)
# ============================================================
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances

RANDOM_STATE = 42
K = 5
SAMPLE_SIZE = 200

def train_test_split_user_based(df, sentiment_filter, test_ratio=0.2, min_items=2):
    train_rows, test_rows = [], []
    for user_id, user_data in df.groupby("UserId"):
        liked = user_data[user_data["Sentiment"] == 1]
        if len(liked) < min_items:
            continue
        split_point = int(len(liked) * (1 - test_ratio))
        test_rows.append(liked.iloc[split_point:])
        if sentiment_filter:
            train_rows.append(liked.iloc[:split_point])
        else:
            test_idx = liked.iloc[split_point:].index
            user_train_pool = user_data.drop(index=test_idx, errors="ignore")
            train_rows.append(user_train_pool)
    train_df = pd.concat(train_rows).reset_index(drop=True)
    test_df = pd.concat(test_rows).reset_index(drop=True)
    return train_df, test_df

def evaluate_at_k(true_items, recommended_items, k):
    recommended_k = recommended_items[:k]
    true_set = set(true_items)
    rec_set = set(recommended_k)
    intersection = true_set.intersection(rec_set)
    precision = len(intersection) / k if k > 0 else 0
    recall = len(intersection) / len(true_set) if len(true_set) > 0 else 0
    return precision, recall

def prepare_content_model(train_df):
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = tfidf.fit_transform(train_df["cleaned_review"])
    return tfidf, tfidf_matrix

def content_based_recommendation(train_df, test_df, user_id, tfidf, tfidf_matrix, k=5):
    user_train = train_df[train_df["UserId"] == user_id]
    user_test = test_df[test_df["UserId"] == user_id]
    if len(user_train) == 0 or len(user_test) == 0:
        return None
    user_vectors = tfidf.transform(user_train["cleaned_review"])
    similarities = cosine_similarity(user_vectors, tfidf_matrix)
    similarity_scores = similarities.mean(axis=0)
    top_indices = similarity_scores.argsort()[::-1][:k]
    recommended_products = train_df.iloc[top_indices]["ProductId"].values
    true_products = user_test["ProductId"].values
    return evaluate_at_k(true_products, recommended_products, k)

def prepare_collaborative_model(train_df, sentiment_filter):
    value_col = "InteractionValue"
    train_df = train_df.copy()
    train_df[value_col] = train_df["Sentiment"] if sentiment_filter else 1
    user_matrix = train_df.pivot_table(index="UserId", columns="ProductId", values=value_col, aggfunc="mean").fillna(0)
    similarity_matrix = 1 - pairwise_distances(user_matrix, metric="cosine")
    return user_matrix, similarity_matrix

def collaborative_filtering(train_df, test_df, user_id, user_matrix, similarity_matrix, k=5):
    if user_id not in user_matrix.index:
        return None
    user_idx = user_matrix.index.get_loc(user_id)
    similar_users = similarity_matrix[user_idx].argsort()[::-1][1:k + 1]
    recommended_products = []
    for idx in similar_users:
        similar_user_id = user_matrix.index[idx]
        products = train_df[train_df["UserId"] == similar_user_id]["ProductId"]
        recommended_products.extend(products)
    recommended_products = list(dict.fromkeys(recommended_products))[:k]
    true_products = test_df[test_df["UserId"] == user_id]["ProductId"].values
    return evaluate_at_k(true_products, recommended_products, k)

def hybrid_recommendation(train_df, test_df, user_id, tfidf, tfidf_matrix, user_matrix, similarity_matrix, alpha=0.9, k=5):
    train_df = train_df.reset_index(drop=True)
    if user_id not in user_matrix.index:
        return None
    user_train = train_df[train_df["UserId"] == user_id]
    user_test = test_df[test_df["UserId"] == user_id]
    if len(user_train) == 0 or len(user_test) == 0:
        return None
    user_vectors = tfidf.transform(user_train["cleaned_review"])
    content_sim = cosine_similarity(user_vectors, tfidf_matrix)
    content_scores = content_sim.mean(axis=0)
    content_scores = (content_scores - content_scores.min()) / (content_scores.max() - content_scores.min() + 1e-9)
    user_idx = user_matrix.index.get_loc(user_id)
    user_sim_vector = similarity_matrix[user_idx]
    collab_scores_products = user_sim_vector @ user_matrix.values
    collab_scores_products = (collab_scores_products - collab_scores_products.min()) / (collab_scores_products.max() - collab_scores_products.min() + 1e-9)
    product_ids = user_matrix.columns
    product_score_map = dict(zip(product_ids, collab_scores_products))
    collab_scores_aligned = np.array([product_score_map.get(pid, 0) for pid in train_df["ProductId"]])
    hybrid_scores = alpha * content_scores + (1 - alpha) * collab_scores_aligned
    top_indices = hybrid_scores.argsort()[::-1][:k]
    recommended_products = train_df.iloc[top_indices]["ProductId"].values
    true_products = user_test["ProductId"].values
    return evaluate_at_k(true_products, recommended_products, k)

def run_condition(df, sentiment_filter, k=K, sample_size=SAMPLE_SIZE, alpha=0.9, seed=RANDOM_STATE):
    train_df, test_df = train_test_split_user_based(df, sentiment_filter=sentiment_filter)
    rng = np.random.RandomState(seed)
    users = train_df["UserId"].unique()
    users = np.array(users)
    rng.shuffle(users)
    if sample_size is not None:
        users = users[:sample_size]
    tfidf, tfidf_matrix = prepare_content_model(train_df)
    user_matrix, similarity_matrix = prepare_collaborative_model(train_df, sentiment_filter)
    pc, rc, pcf, rcf, ph, rh = [], [], [], [], [], []
    for user_id in users:
        rc_res = content_based_recommendation(train_df, test_df, user_id, tfidf, tfidf_matrix, k)
        cf_res = collaborative_filtering(train_df, test_df, user_id, user_matrix, similarity_matrix, k)
        h_res = hybrid_recommendation(train_df, test_df, user_id, tfidf, tfidf_matrix, user_matrix, similarity_matrix, alpha, k)
        if rc_res is not None: pc.append(rc_res[0]); rc.append(rc_res[1])
        if cf_res is not None: pcf.append(cf_res[0]); rcf.append(cf_res[1])
        if h_res is not None: ph.append(h_res[0]); rh.append(h_res[1])
    return {
        "n_train": len(train_df), "n_test": len(test_df), "n_users": len(users),
        "Content_P@5": np.mean(pc) if pc else np.nan, "Content_R@5": np.mean(rc) if rc else np.nan,
        "Collab_P@5": np.mean(pcf) if pcf else np.nan, "Collab_R@5": np.mean(rcf) if rcf else np.nan,
        "Hybrid_P@5": np.mean(ph) if ph else np.nan, "Hybrid_R@5": np.mean(rh) if rh else np.nan,
    }

print("=== KOŞUL A: Duygu filtresi VAR ===")
res_with = run_condition(filtered_data, sentiment_filter=True)
print(res_with)

print("\n=== KOŞUL B: Duygu filtresi YOK ===")
res_without = run_condition(filtered_data, sentiment_filter=False)
print(res_without)

comp = pd.DataFrame([
    {"Yöntem": "İçerik Tabanlı", "P@5 (VAR)": res_with["Content_P@5"], "P@5 (YOK)": res_without["Content_P@5"],
     "R@5 (VAR)": res_with["Content_R@5"], "R@5 (YOK)": res_without["Content_R@5"]},
    {"Yöntem": "İşbirlikçi", "P@5 (VAR)": res_with["Collab_P@5"], "P@5 (YOK)": res_without["Collab_P@5"],
     "R@5 (VAR)": res_with["Collab_R@5"], "R@5 (YOK)": res_without["Collab_R@5"]},
    {"Yöntem": "Hibrit", "P@5 (VAR)": res_with["Hybrid_P@5"], "P@5 (YOK)": res_without["Hybrid_P@5"],
     "R@5 (VAR)": res_with["Hybrid_R@5"], "R@5 (YOK)": res_without["Hybrid_R@5"]},
])
print("\n=== KARŞILAŞTIRMA ===")
print(comp.to_string(index=False))
comp.to_csv("ablation_sonuclari.csv", index=False)
print("\nKaydedildi: ablation_sonuclari.csv (Colab dosya panelinden indirebilirsin)")
