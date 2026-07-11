"""
Duygu-filtreli vs duygu-filtresiz öneri sistemi ablation deneyi.

Amaç: Tezdeki (23022026.ipynb) içerik tabanlı / işbirlikçi / hibrit öneri
algoritmalarını AYNI KOD, AYNI Precision@K/Recall@K metodolojisiyle,
yalnızca "kullanıcı profili ve etkileşim matrisi kurulurken duygu
filtresi uygulanıyor mu uygulanmıyor mu" değişkenini kontrollü şekilde
değiştirerek çalıştırmak.

Ground truth (test seti) HER İKİ KOŞULDA DA aynı tanımla kalıyor:
kullanıcının gerçekten olumlu (Sentiment==1) değerlendirdiği, elde
tutulan (held-out) ürünler. Böylece iki koşulun ölçtüğü şey birebir
aynı soru oluyor: "Bu yöntem, kullanıcının GERÇEKTEN BEĞENECEĞİ
ürünleri ne kadar iyi buluyor?" -- tek fark, modelin bunu tahmin
ederken duygu bilgisinden yararlanıp yararlanmadığı.
"""

import string
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances

RANDOM_STATE = 42
K = 5
SAMPLE_SIZE = 200  # tezdeki notebook ile aynı örnek boyutu

# ---------------------------------------------------------------------
# 1) Veri hazırlığı (prepare_data'nın basitleştirilmiş, nltk'siz hali)
# ---------------------------------------------------------------------
PUNCT_TABLE = str.maketrans('', '', string.punctuation)


def clean_text(text):
    text = str(text).lower()
    text = text.translate(PUNCT_TABLE)
    return text


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df[['Text', 'Score', 'ProductId', 'UserId']].copy()
    df = df[df['Score'] != 3]                      # tezdeki gibi nötr puan atılıyor
    df['Sentiment'] = df['Score'].apply(lambda x: 1 if x >= 4 else 0)
    df['cleaned_review'] = df['Text'].apply(clean_text)
    df = df.dropna(subset=['cleaned_review', 'UserId', 'ProductId'])
    return df


# ---------------------------------------------------------------------
# 2) Train/test kurulumu -- SENTIMENT_FILTER parametresiyle kontrol edilir
# ---------------------------------------------------------------------
def train_test_split_user_based(df, sentiment_filter, test_ratio=0.2, min_items=2):
    """
    sentiment_filter=True  -> tezdeki mevcut yaklaşım: yalnızca
                               Sentiment==1 etkileşimler kullanılır
                               (hem train hem test).
    sentiment_filter=False -> "duygusuz" temel model: kullanıcının TÜM
                               etkileşimleri (olumlu/olumsuz) kullanılır;
                               model duygu bilgisinden habersizdir.
    Test seti HER ZAMAN yalnızca gerçekten sevilen (Sentiment==1)
    ürünlerden kurulur -- başarı ölçütü sabit tutulur.
    """
    train_rows, test_rows = [], []

    for user_id, user_data in df.groupby('UserId'):
        liked = user_data[user_data['Sentiment'] == 1]
        if len(liked) < min_items:
            continue

        split_point = int(len(liked) * (1 - test_ratio))
        test_rows.append(liked.iloc[split_point:])

        if sentiment_filter:
            train_rows.append(liked.iloc[:split_point])
        else:
            # test'e ayrılan satırları train'den çıkar ki sızıntı olmasın,
            # geri kalan TÜM etkileşimleri (olumlu+olumsuz) kullan
            test_idx = liked.iloc[split_point:].index
            user_train_pool = user_data.drop(index=test_idx, errors='ignore')
            train_rows.append(user_train_pool)

    train_df = pd.concat(train_rows).reset_index(drop=True)
    test_df = pd.concat(test_rows).reset_index(drop=True)
    return train_df, test_df


# ---------------------------------------------------------------------
# 3) Değerlendirme metriği (tezdekiyle birebir aynı)
# ---------------------------------------------------------------------
def evaluate_at_k(true_items, recommended_items, k):
    recommended_k = recommended_items[:k]
    true_set = set(true_items)
    rec_set = set(recommended_k)
    intersection = true_set.intersection(rec_set)
    precision = len(intersection) / k if k > 0 else 0
    recall = len(intersection) / len(true_set) if len(true_set) > 0 else 0
    return precision, recall


# ---------------------------------------------------------------------
# 4) İçerik tabanlı, işbirlikçi, hibrit -- tezdeki fonksiyonların aynısı
#    (matrisin değeri sentiment_filter=False durumunda "etkileşim var mı"
#    (1/0) olacak şekilde ayarlanır; sentiment_filter=True durumunda
#    zaten yalnızca Sentiment==1 satırlar var, değer hep 1)
# ---------------------------------------------------------------------
def prepare_content_model(train_df):
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(train_df['cleaned_review'])
    return tfidf, tfidf_matrix


def content_based_recommendation(train_df, test_df, user_id, tfidf, tfidf_matrix, k=5):
    user_train = train_df[train_df['UserId'] == user_id]
    user_test = test_df[test_df['UserId'] == user_id]
    if len(user_train) == 0 or len(user_test) == 0:
        return None
    user_vectors = tfidf.transform(user_train['cleaned_review'])
    similarities = cosine_similarity(user_vectors, tfidf_matrix)
    similarity_scores = similarities.mean(axis=0)
    top_indices = similarity_scores.argsort()[::-1][:k]
    recommended_products = train_df.iloc[top_indices]['ProductId'].values
    true_products = user_test['ProductId'].values
    return evaluate_at_k(true_products, recommended_products, k)


def prepare_collaborative_model(train_df, sentiment_filter, collab_user_ids=None):
    value_col = 'InteractionValue'
    train_df = train_df.copy()
    if sentiment_filter:
        train_df[value_col] = train_df['Sentiment']  # zaten hep 1
    else:
        train_df[value_col] = 1  # yalnızca "etkileşim var mı" -- duygu bilgisi YOK

    user_matrix = train_df.pivot_table(
        index='UserId', columns='ProductId', values=value_col, aggfunc='mean'
    ).fillna(0)

    # Büyük kullanıcı sayısında pairwise_distances O(n^2) patlıyor; makul bir üst
    # sınır koyuyoruz. ÖNEMLİ: hangi kullanıcıların matrise gireceği HER İKİ
    # koşulda (duygu filtreli / filtresiz) da AYNI sabit kullanıcı kümesinden
    # (collab_user_ids) seçilir -- yoksa alfabetik ilk-N seçimi iki koşulda farklı
    # kullanıcı setleri getirebilir ve karşılaştırma adil olmaz.
    if collab_user_ids is not None:
        present = [u for u in collab_user_ids if u in user_matrix.index]
        user_matrix = user_matrix.loc[present]

    similarity_matrix = 1 - pairwise_distances(user_matrix, metric='cosine')
    return user_matrix, similarity_matrix


def collaborative_filtering(train_df, test_df, user_id, user_matrix, similarity_matrix, k=5):
    if user_id not in user_matrix.index:
        return None
    user_idx = user_matrix.index.get_loc(user_id)
    similar_users = similarity_matrix[user_idx].argsort()[::-1][1:k + 1]
    recommended_products = []
    for idx in similar_users:
        similar_user_id = user_matrix.index[idx]
        products = train_df[train_df['UserId'] == similar_user_id]['ProductId']
        recommended_products.extend(products)
    recommended_products = list(dict.fromkeys(recommended_products))[:k]
    true_products = test_df[test_df['UserId'] == user_id]['ProductId'].values
    return evaluate_at_k(true_products, recommended_products, k)


def hybrid_recommendation(train_df, test_df, user_id, tfidf, tfidf_matrix,
                           user_matrix, similarity_matrix, alpha=0.9, k=5):
    train_df = train_df.reset_index(drop=True)
    if user_id not in user_matrix.index:
        return None
    user_train = train_df[train_df['UserId'] == user_id]
    user_test = test_df[test_df['UserId'] == user_id]
    if len(user_train) == 0 or len(user_test) == 0:
        return None

    user_vectors = tfidf.transform(user_train['cleaned_review'])
    content_sim = cosine_similarity(user_vectors, tfidf_matrix)
    content_scores = content_sim.mean(axis=0)
    content_scores = (content_scores - content_scores.min()) / (
        content_scores.max() - content_scores.min() + 1e-9)

    user_idx = user_matrix.index.get_loc(user_id)
    user_sim_vector = similarity_matrix[user_idx]
    collab_scores_products = user_sim_vector @ user_matrix.values
    collab_scores_products = (collab_scores_products - collab_scores_products.min()) / (
        collab_scores_products.max() - collab_scores_products.min() + 1e-9)

    product_ids = user_matrix.columns
    product_score_map = dict(zip(product_ids, collab_scores_products))
    collab_scores_aligned = np.array(
        [product_score_map.get(pid, 0) for pid in train_df['ProductId']])

    hybrid_scores = alpha * content_scores + (1 - alpha) * collab_scores_aligned
    top_indices = hybrid_scores.argsort()[::-1][:k]
    recommended_products = train_df.iloc[top_indices]['ProductId'].values
    true_products = user_test['ProductId'].values
    return evaluate_at_k(true_products, recommended_products, k)


# ---------------------------------------------------------------------
# 5) Tüm kullanıcılar üzerinden ortalama
# ---------------------------------------------------------------------
def get_eligible_users(df, min_items=2):
    """sentiment_filter True/False fark etmeksizin AYNI kullanıcı evrenini
    tanımlar (train_test_split_user_based'teki eleme kuralıyla birebir aynı)."""
    liked_counts = df[df['Sentiment'] == 1].groupby('UserId').size()
    return liked_counts[liked_counts >= min_items].index.to_numpy()


def run_condition(df, sentiment_filter, k=K, sample_size=SAMPLE_SIZE, alpha=0.9,
                   seed=RANDOM_STATE, eval_user_ids=None, collab_user_ids=None):
    train_df, test_df = train_test_split_user_based(df, sentiment_filter=sentiment_filter)

    if eval_user_ids is not None:
        users = eval_user_ids
    else:
        rng = np.random.RandomState(seed)
        users = train_df['UserId'].unique()
        rng.shuffle(users)
        if sample_size is not None:
            users = users[:sample_size]

    tfidf, tfidf_matrix = prepare_content_model(train_df)
    user_matrix, similarity_matrix = prepare_collaborative_model(
        train_df, sentiment_filter, collab_user_ids=collab_user_ids)

    pc, rc, pcf, rcf, ph, rh = [], [], [], [], [], []

    for user_id in users:
        rc_res = content_based_recommendation(train_df, test_df, user_id, tfidf, tfidf_matrix, k)
        cf_res = collaborative_filtering(train_df, test_df, user_id, user_matrix, similarity_matrix, k)
        h_res = hybrid_recommendation(train_df, test_df, user_id, tfidf, tfidf_matrix,
                                       user_matrix, similarity_matrix, alpha, k)
        if rc_res is not None:
            pc.append(rc_res[0]); rc.append(rc_res[1])
        if cf_res is not None:
            pcf.append(cf_res[0]); rcf.append(cf_res[1])
        if h_res is not None:
            ph.append(h_res[0]); rh.append(h_res[1])

    return {
        'n_train_interactions': len(train_df),
        'n_test_interactions': len(test_df),
        'n_users_evaluated': len(users),
        'Content_Precision@5': np.mean(pc) if pc else np.nan,
        'Content_Recall@5': np.mean(rc) if rc else np.nan,
        'Collab_Precision@5': np.mean(pcf) if pcf else np.nan,
        'Collab_Recall@5': np.mean(rcf) if rcf else np.nan,
        'Hybrid_Precision@5': np.mean(ph) if ph else np.nan,
        'Hybrid_Recall@5': np.mean(rh) if rh else np.nan,
    }


if __name__ == '__main__':
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'sampled_reviews.csv'
    df = load_data(csv_path)
    print(f'Veri seti (Score!=3 sonrası): {df.shape}')
    print(df['Sentiment'].value_counts())

    print('\n=== KOŞUL A: Duygu filtresi VAR (tezdeki mevcut yaklaşım) ===')
    res_with = run_condition(df, sentiment_filter=True)
    for k_, v in res_with.items():
        print(f'  {k_}: {v}')

    print('\n=== KOŞUL B: Duygu filtresi YOK (baseline) ===')
    res_without = run_condition(df, sentiment_filter=False)
    for k_, v in res_without.items():
        print(f'  {k_}: {v}')

    print('\n=== KARŞILAŞTIRMA TABLOSU ===')
    rows = []
    for method in ['Content', 'Collab', 'Hybrid']:
        rows.append({
            'Yöntem': method,
            'Precision@5 (duygu VAR)': res_with[f'{method}_Precision@5'],
            'Precision@5 (duygu YOK)': res_without[f'{method}_Precision@5'],
            'Recall@5 (duygu VAR)': res_with[f'{method}_Recall@5'],
            'Recall@5 (duygu YOK)': res_without[f'{method}_Recall@5'],
        })
    comp = pd.DataFrame(rows)
    print(comp.to_string(index=False))
    comp.to_csv('/tmp/repo_check/ablation_results.csv', index=False)
    print('\nKaydedildi: /tmp/repo_check/ablation_results.csv')
