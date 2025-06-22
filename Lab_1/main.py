import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
import random

random.seed(0)
np.random.seed(0)

def load_data():    
    train_data = pd.read_csv('data/review.csv')
    valid_data = pd.read_csv('data/validation.csv')
    test_data = pd.read_csv('data/prediction.csv')
    
    return train_data, valid_data, test_data

def create_matrices(train_data):
    user_item_matrix = train_data.pivot_table(index='ReviewerID', columns='ProductID', values='Star')
    
    global_mean = train_data['Star'].mean()    
    user_means = user_item_matrix.mean(axis=1).fillna(global_mean)
    item_means = user_item_matrix.mean(axis=0).fillna(global_mean)
    
    user_item_matrix_filled = user_item_matrix.fillna(global_mean)
    
    return {
        'user_item': user_item_matrix,
        'item_user': user_item_matrix.T,
        'user_item_filled': user_item_matrix_filled,
        'user_means': user_means,
        'item_means': item_means,
        'global_mean': global_mean
    }

def compute_similarities(matrices):
    # User similarity 
    user_counts = (matrices['user_item'] > 0).sum(axis=1)
    user_sim = cosine_similarity(matrices['user_item_filled'])
    user_sim = user_sim * np.minimum(1, np.log(user_counts.values)/np.log(10))  
    
    # Item similarity 
    item_counts = (matrices['user_item'] > 0).sum(axis=0)
    item_sim = cosine_similarity(matrices['user_item_filled'].T)
    item_sim = item_sim * (50 / (50 + item_counts.values)) 
    
    return (
        pd.DataFrame(user_sim, index=matrices['user_item_filled'].index, columns=matrices['user_item_filled'].index),
        pd.DataFrame(item_sim, index=matrices['user_item_filled'].columns, columns=matrices['user_item_filled'].columns)
    )


def predict_user_based(user_id, item_id, matrices, user_sim_df, k=20):
    if user_id not in matrices['user_item'].index or item_id not in matrices['user_item'].columns:
        return matrices['global_mean']
    
    # Similar users that rated the item
    user_mean = matrices['user_means'][user_id]
    similar_users = user_sim_df[user_id].sort_values(ascending=False)[1:k+1]  
    
    # Ratings from similar users
    item_ratings = matrices['user_item'][item_id]
    valid_users = similar_users.index.intersection(item_ratings.dropna().index)
    
    if len(valid_users) == 0:
        return user_mean
    
    # Weighted average
    numerator = (similar_users[valid_users] * (item_ratings[valid_users] - matrices['user_means'][valid_users])).sum()
    denominator = similar_users[valid_users].abs().sum()
    
    if denominator == 0:
        return user_mean
    
    return user_mean + numerator / denominator

def predict_item_based(user_id, item_id, matrices, item_sim_df, k=20):
    if item_id not in matrices['item_user'].index or user_id not in matrices['item_user'].columns:
        return matrices['global_mean']
    
    # Similar items
    item_mean = matrices['item_means'][item_id]
    similar_items = item_sim_df[item_id].sort_values(ascending=False)[1:k+1] 
    
    # User's ratings
    user_ratings = matrices['item_user'][user_id]
    valid_items = similar_items.index.intersection(user_ratings.dropna().index)
    
    if len(valid_items) == 0:
        return item_mean if not np.isnan(item_mean) else matrices['global_mean']
    
    # Weighted average
    numerator = (similar_items[valid_items] * user_ratings[valid_items]).sum()
    denominator = similar_items[valid_items].abs().sum()
    
    if denominator == 0:
        return item_mean if not np.isnan(item_mean) else matrices['global_mean']
    
    return numerator / denominator

def svd_predict(matrices, n_components=100):
    users = matrices['user_item'].index
    items = matrices['user_item'].columns
    user_map = {u: i for i, u in enumerate(users)}
    item_map = {p: i for i, p in enumerate(items)}
    
    # Mean centering
    rows, cols, data = [], [], []
    for (user, item), rating in matrices['user_item'].stack().items():
        rows.append(user_map[user])
        cols.append(item_map[item])
        data.append(rating - matrices['global_mean'] - 
                   (matrices['user_means'][user] - matrices['global_mean']) - 
                   (matrices['item_means'][item] - matrices['global_mean']))
    
    sparse_mat = csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))
    
    U, sigma, Vt = svds(sparse_mat, k=n_components)
    sigma = np.diag(sigma)
    pred = np.dot(np.dot(U, sigma), Vt)    
    pred += matrices['global_mean']
    pred += (matrices['user_means'].values - matrices['global_mean']).reshape(-1, 1)
    pred += (matrices['item_means'].values - matrices['global_mean'])
    
    return pd.DataFrame(pred, index=users, columns=items)

def hybrid_predict(user_id, item_id, matrices, user_sim_df, item_sim_df, svd_pred_df):
    try:
        user_pred = predict_user_based(user_id, item_id, matrices, user_sim_df)
    except:
        user_pred = matrices['global_mean']
    try:
        item_pred = predict_item_based(user_id, item_id, matrices, item_sim_df)
    except:
        item_pred = matrices['global_mean']
    try:
        svd_pred = svd_pred_df.loc[user_id, item_id]
    except:
        svd_pred = matrices['global_mean']

    # Starting weights    
    user_weight = 0.3
    item_weight = 0.3
    svd_weight = 0.4
    
    # Adjust weights based on number of ratings
    user_ratings_count = (matrices['user_item'].loc[user_id].notna()).sum() if user_id in matrices['user_item'].index else 0
    item_ratings_count = (matrices['item_user'].loc[item_id].notna()).sum() if item_id in matrices['item_user'].index else 0
    
    if user_ratings_count < 5:
        user_weight *= 0.5
        svd_weight += 0.1
    if item_ratings_count < 5:
        item_weight *= 0.5
        svd_weight += 0.1
    
    total_weight = user_weight + item_weight + svd_weight
    pred = (
        user_weight * np.clip(user_pred, 1, 5) +
        item_weight * np.clip(item_pred, 1, 5) +
        svd_weight * np.clip(svd_pred, 1, 5)
    ) / total_weight
    
    return pred

def main():
    train_data, valid_data, test_data = load_data()

    matrices = create_matrices(train_data)
    user_sim_df, item_sim_df = compute_similarities(matrices)
    svd_pred_df = svd_predict(matrices, n_components=100)
    
    valid_predictions = []
    for _, row in valid_data.iterrows():
        pred = hybrid_predict(row['ReviewerID'], row['ProductID'], 
                            matrices, user_sim_df, item_sim_df, svd_pred_df)
        valid_predictions.append(pred)
    
    test_predictions = []
    for _, row in test_data.iterrows():
        pred = hybrid_predict(row['ReviewerID'], row['ProductID'],
                            matrices, user_sim_df, item_sim_df, svd_pred_df)
        test_predictions.append(pred)
    
    valid_data[['ReviewerID', 'ProductID']].assign(Star=valid_predictions)\
        .to_csv('validation_prediction.csv', index=False)
    
    test_data[['ReviewerID', 'ProductID']].assign(Star=test_predictions)\
        .to_csv('data/prediction.csv', index=False)
    
    print("Test predictions saved to prediction.csv")
    print("Validation predictions saved to validation_prediction.csv")

if __name__ == "__main__":
    main()