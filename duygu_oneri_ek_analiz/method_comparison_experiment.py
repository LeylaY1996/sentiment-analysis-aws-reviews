"""
Farklı duygu analizi yöntemlerinin (SVM, Naive Bayes, Lojistik Regresyon,
Sözlük Tabanlı) öneri sistemi performansına etkisinin karşılaştırılması.

Mantık: Her yöntem, yorum METNİNDEN duygu tahmini üretir (Sentiment_true'yu
GÖRMEDEN -- out-of-fold cross-val tahminleri kullanılır, sızıntı önlenir).
Bu tahmin edilen etiket, öneri sisteminin kullanıcı profilini/etkileşim
matrisini kurmak için "beğenilen ürün" filtresi olarak kullanılır.
Test seti (başarı ölçütü) HER YÖNTEMDE AYNI kalır: kullanıcının GERÇEKTEN
(Score bazlı) sevdiği, elde tutulan ürünler. Böylece "hangi duygu analizi
yöntemiyle etiketlenirse öneri sistemi daha iyi çalışır" sorusu adil
biçimde test edilir.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances

# Tezdeki 3.4.2.5 / kod hücresi 21 ile aynı hedef tabanlı (aspect-based) öznitelikler
ASPECT_KEYWORDS = {
    'hız': ['fast', 'quick', 'speed', 'late', 'delayed'],
    'kalite': ['good', 'great', 'excellent', 'poor', 'bad', 'terrible', 'quality'],
}


def extract_aspect_features(texts):
    rows = []
    for t in texts:
        words = t.split()
        rows.append([sum(w in kws for w in words) for kws in ASPECT_KEYWORDS.values()])
    return np.array(rows)

from ablation_experiment import (
    load_data, evaluate_at_k, prepare_content_model, content_based_recommendation,
    hybrid_recommendation,
)

RANDOM_STATE = 42
K = 5


# ---------------------------------------------------------------------
# Lexicon tabanlı (tezdeki 3.4.2.7 / kod hücresi 20 ile aynı mantık)
# ---------------------------------------------------------------------
POS_WORDS = {'good', 'great', 'excellent', 'awesome', 'nice', 'love', 'best', 'delicious', 'perfect'}
NEG_WORDS = {'bad', 'terrible', 'poor', 'worst', 'awful', 'disappointing', 'horrible', 'stale'}


def lexicon_predict(texts):
    preds = []
    for t in texts:
        words = t.split()
        score = sum(1 for w in words if w in POS_WORDS) - sum(1 for w in words if w in NEG_WORDS)
        preds.append(1 if score > 0 else 0)
    return np.array(preds)


# ---------------------------------------------------------------------
# Out-of-fold duygu tahminleri (her yöntem için)
# ---------------------------------------------------------------------
def get_oof_predictions(df, n_splits=5, seed=RANDOM_STATE):
    texts = df['cleaned_review'].values
    y_true = df['Sentiment'].values

    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(texts)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    results = {}
    accs = {}

    models = {
        'SVM': LinearSVC(),
        'Naive Bayes': MultinomialNB(),
        'Lojistik Regresyon': LogisticRegression(max_iter=1000, random_state=seed),
        'Yapay Sinir Ağı (MLP)': MLPClassifier(
            hidden_layer_sizes=(64,), max_iter=100, early_stopping=True, random_state=seed
        ),
    }

    for name, model in models.items():
        preds = cross_val_predict(model, X_tfidf, y_true, cv=skf, n_jobs=-1)
        results[name] = preds
        accs[name] = {
            'accuracy': accuracy_score(y_true, preds),
            'f1_macro': f1_score(y_true, preds, average='macro'),
        }

    lex_preds = lexicon_predict(texts)
    results['Sözlük Tabanlı'] = lex_preds
    accs['Sözlük Tabanlı'] = {
        'accuracy': accuracy_score(y_true, lex_preds),
        'f1_macro': f1_score(y_true, lex_preds, average='macro'),
    }

    # Hedef Tabanlı (Aspect-Based) -- yalnızca scikit-learn gerektirir
    X_aspect = extract_aspect_features(texts)
    aspect_preds = cross_val_predict(
        RandomForestClassifier(random_state=seed), X_aspect, y_true, cv=skf, n_jobs=-1
    )
    results['Hedef Tabanlı'] = aspect_preds
    accs['Hedef Tabanlı'] = {
        'accuracy': accuracy_score(y_true, aspect_preds),
        'f1_macro': f1_score(y_true, aspect_preds, average='macro'),
    }

    return results, accs


# ---------------------------------------------------------------------
# Öneri sistemi: train/test kurulumu, train_label_col'a göre "beğenilen" filtre
# ---------------------------------------------------------------------
def train_test_split_by_label(df, train_label_col, test_ratio=0.2, min_items=2):
    train_rows, test_rows = [], []
    for user_id, user_data in df.groupby('UserId'):
        liked_true = user_data[user_data['Sentiment'] == 1]
        if len(liked_true) < min_items:
            continue
        split_point = int(len(liked_true) * (1 - test_ratio))
        test_rows.append(liked_true.iloc[split_point:])

        test_idx = liked_true.iloc[split_point:].index
        candidate_pool = user_data.drop(index=test_idx, errors='ignore')
        train_liked = candidate_pool[candidate_pool[train_label_col] == 1]
        if len(train_liked) == 0:
            continue  # bu yöntemle bu kullanıcı için hiç "beğenilen" ürün öngörülemedi
        train_rows.append(train_liked)

    train_df = pd.concat(train_rows).reset_index(drop=True)
    test_df = pd.concat(test_rows).reset_index(drop=True)
    return train_df, test_df


def prepare_collaborative_model(train_df, value_col, collab_user_ids=None):
    train_df = train_df.copy()
    train_df['_val'] = 1  # zaten yalnızca "beğenilen" (etiketi 1 olan) satırlar var
    user_matrix = train_df.pivot_table(
        index='UserId', columns='ProductId', values='_val', aggfunc='mean'
    ).fillna(0)
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


def run_method_condition(df, train_label_col, eval_user_ids, collab_user_ids, k=K, alpha=0.9):
    train_df, test_df = train_test_split_by_label(df, train_label_col)

    tfidf, tfidf_matrix = prepare_content_model(train_df)
    user_matrix, similarity_matrix = prepare_collaborative_model(train_df, train_label_col, collab_user_ids)

    pc, rc, pcf, rcf, ph, rh = [], [], [], [], [], []

    for user_id in eval_user_ids:
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
        'n_train': len(train_df), 'n_test': len(test_df),
        'Content_P@5': np.mean(pc) if pc else np.nan, 'Content_R@5': np.mean(rc) if rc else np.nan,
        'Collab_P@5': np.mean(pcf) if pcf else np.nan, 'Collab_R@5': np.mean(rcf) if rcf else np.nan,
        'Hybrid_P@5': np.mean(ph) if ph else np.nan, 'Hybrid_R@5': np.mean(rh) if rh else np.nan,
    }


if __name__ == '__main__':
    import sys, json, time
    t0 = time.time()
    csv_path = sys.argv[1] if len(sys.argv) > 1 else '/mnt/user-data/uploads/Downloads/reviews_med.csv'
    df = load_data(csv_path)
    print('Veri:', df.shape, flush=True)

    print('=== Out-of-fold duygu tahminleri üretiliyor ===', flush=True)
    preds, accs = get_oof_predictions(df)
    for name, a in accs.items():
        print(f'  {name}: accuracy={a["accuracy"]:.4f}  f1_macro={a["f1_macro"]:.4f}', flush=True)
    print('süre:', time.time() - t0, flush=True)

    for name, p in preds.items():
        col = f'Pred_{name}'
        df[col] = p

    from ablation_experiment import get_eligible_users
    elig = get_eligible_users(df, min_items=2)
    rng = np.random.RandomState(RANDOM_STATE)
    elig_shuffled = elig.copy()
    rng.shuffle(elig_shuffled)
    eval_users = elig_shuffled[:300]
    collab_users = elig_shuffled[:1200]

    method_cols = {
        'Gerçek Etiket (Score bazlı)': 'Sentiment',
        'SVM': 'Pred_SVM',
        'Naive Bayes': 'Pred_Naive Bayes',
        'Lojistik Regresyon': 'Pred_Lojistik Regresyon',
        'Sözlük Tabanlı': 'Pred_Sözlük Tabanlı',
        'Hedef Tabanlı': 'Pred_Hedef Tabanlı',
        'Yapay Sinir Ağı (MLP)': 'Pred_Yapay Sinir Ağı (MLP)',
    }

    all_results = {}
    for label, col in method_cols.items():
        print(f'=== Öneri sistemi çalıştırılıyor: {label} ===', flush=True)
        res = run_method_condition(df, col, eval_users, collab_users)
        all_results[label] = res
        for k_, v in res.items():
            print(' ', k_, v, flush=True)
        print('ara süre:', time.time() - t0, flush=True)

    json.dump({'classification_metrics': accs, 'recommender_results': all_results},
               open('/tmp/repo_check/method_comparison_results.json', 'w'), indent=2)
    print('\nKaydedildi: method_comparison_results.json')
    print('toplam süre:', time.time() - t0, 'sn')
