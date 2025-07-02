from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

DATA_FILE = 'products_1000.csv'
DATA_DIR = os.path.join(os.path.dirname(__file__), 'Data')

class RecommendationAPI:
    def __init__(self):
        self.user_item_matrix = None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.content_similarity_matrix = None
        self.knn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
        self.decision_tree = DecisionTreeClassifier(random_state=42)
        self.is_trained = False
        
    def initialize_data(self):
        """Initialize with sample data"""
        np.random.seed(42)
        
        # Generate sample data (same as main system)
        users = [f'user_{i}' for i in range(1, 101)]
        products = [f'product_{i}' for i in range(1, 51)]
        
        # Product descriptions
        descriptions = [
            'smartphone with high resolution camera and fast processor',
            'laptop computer with SSD storage and long battery life',
            'wireless headphones with noise cancellation',
            'smart watch with fitness tracking features',
            'tablet with touch screen and wifi connectivity',
            'cotton t-shirt comfortable casual wear',
            'denim jeans classic blue style',
            'running shoes lightweight breathable',
            'winter jacket warm waterproof',
            'dress shirt formal business attire',
            'mystery novel thriller suspense story',
            'science fiction space adventure',
            'romance novel love story',
            'biography historical figure',
            'cookbook recipes healthy cooking',
            'coffee maker automatic brewing',
            'vacuum cleaner powerful suction',
            'bedding sheets soft comfortable',
            'kitchen knife sharp stainless steel',
            'lamp LED bright lighting',
            'yoga mat non-slip exercise',
            'tennis racket lightweight carbon',
            'football leather official size',
            'bicycle mountain bike durable',
            'dumbbells weight training equipment'
        ] * 2  # Repeat to get 50 products
        
        categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports'] * 10
        
        # Generate ratings data
        ratings_data = []
        for user in users[:50]:
            n_ratings = np.random.randint(10, 21)
            user_products = np.random.choice(products, n_ratings, replace=False)
            for product in user_products:
                rating = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.1, 0.2, 0.3, 0.3])
                ratings_data.append({
                    'user_id': user,
                    'product_id': product,
                    'rating': rating
                })
        
        # Create DataFrames
        self.ratings_df = pd.DataFrame(ratings_data)
        self.products_df = pd.DataFrame({
            'product_id': products,
            'description': descriptions,
            'category': categories
        })
        
        # Preprocess data
        self.user_item_matrix = self.ratings_df.pivot_table(
            index='user_id', 
            columns='product_id', 
            values='rating'
        ).fillna(0)
        
        # Content-based features
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.products_df['description'])
        self.content_similarity_matrix = cosine_similarity(tfidf_matrix)
        
        self.is_trained = True
        
    def get_recommendations(self, user_id, method='hybrid', n_recommendations=5):
        """Get recommendations for a user"""
        if not self.is_trained:
            return []
            
        if method == 'collaborative':
            return self.collaborative_filtering(user_id, n_recommendations)
        elif method == 'content':
            return self.content_based_recommendations(user_id, n_recommendations)
        else:  # hybrid
            return self.hybrid_recommendations(user_id, n_recommendations)
    
    def collaborative_filtering(self, user_id, n_recommendations=5):
        """Collaborative filtering recommendations"""
        if user_id not in self.user_item_matrix.index:
            return []
        
        self.knn_model.fit(self.user_item_matrix.values)
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        distances, indices = self.knn_model.kneighbors([self.user_item_matrix.iloc[user_idx].values])
        
        similar_users = [self.user_item_matrix.index[i] for i in indices[0][1:]]
        user_ratings = self.user_item_matrix.loc[user_id]
        unrated_items = user_ratings[user_ratings == 0].index
        
        recommendations = {}
        for item in unrated_items:
            score = 0
            count = 0
            for similar_user in similar_users:
                if self.user_item_matrix.loc[similar_user, item] > 0:
                    score += self.user_item_matrix.loc[similar_user, item]
                    count += 1
            if count > 0:
                recommendations[item] = score / count
        
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return [item for item, score in sorted_recommendations[:n_recommendations]]
    
    def content_based_recommendations(self, user_id, n_recommendations=5):
        """Content-based recommendations"""
        user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
        high_rated_items = user_ratings[user_ratings['rating'] >= 4]['product_id'].tolist()
        
        if not high_rated_items:
            return []
        
        # Use the most recent highly rated item
        seed_product = high_rated_items[0]
        product_idx = self.products_df[self.products_df['product_id'] == seed_product].index[0]
        
        similarity_scores = list(enumerate(self.content_similarity_matrix[product_idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        similar_products = similarity_scores[1:n_recommendations+1]
        recommendations = [self.products_df.iloc[i[0]]['product_id'] for i in similar_products]
        
        # Filter out already rated items
        user_rated_items = user_ratings['product_id'].tolist()
        recommendations = [item for item in recommendations if item not in user_rated_items]
        
        return recommendations[:n_recommendations]
    
    def hybrid_recommendations(self, user_id, n_recommendations=5):
        """Hybrid recommendations"""
        collab_recs = self.collaborative_filtering(user_id, n_recommendations)
        content_recs = self.content_based_recommendations(user_id, n_recommendations)
        
        # Combine and deduplicate
        all_recs = list(set(collab_recs + content_recs))
        return all_recs[:n_recommendations]
    
    def get_product_details(self, product_ids):
        """Get product details for given IDs"""
        products = []
        for product_id in product_ids:
            product_info = self.products_df[self.products_df['product_id'] == product_id]
            if not product_info.empty:
                products.append({
                    'id': product_id,
                    'description': product_info.iloc[0]['description'],
                    'category': product_info.iloc[0]['category']
                })
        return products

# Initialize the recommendation system
rec_api = RecommendationAPI()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/initialize', methods=['POST'])
def initialize():
    """Initialize the recommendation system"""
    try:
        rec_api.initialize_data()
        return jsonify({
            'status': 'success',
            'message': 'Recommendation system initialized successfully',
            'total_users': len(rec_api.ratings_df['user_id'].unique()),
            'total_products': len(rec_api.products_df),
            'total_ratings': len(rec_api.ratings_df)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/users')
def get_users():
    """Get list of available users"""
    if not rec_api.is_trained:
        return jsonify({'error': 'System not initialized'})
    
    users = rec_api.ratings_df['user_id'].unique().tolist()
    return jsonify({'users': users})

@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    """Get recommendations for a user"""
    if not rec_api.is_trained:
        return jsonify({'error': 'System not initialized'})
    
    data = request.json
    user_id = data.get('user_id')
    method = data.get('method', 'hybrid')
    n_recommendations = data.get('n_recommendations', 5)
    
    if not user_id:
        return jsonify({'error': 'User ID is required'})
    
    try:
        recommendations = rec_api.get_recommendations(user_id, method, n_recommendations)
        products = rec_api.get_product_details(recommendations)
        
        return jsonify({
            'user_id': user_id,
            'method': method,
            'recommendations': products
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/user_stats/<user_id>')
def get_user_stats(user_id):
    """Get user statistics"""
    if not rec_api.is_trained:
        return jsonify({'error': 'System not initialized'})
    
    user_ratings = rec_api.ratings_df[rec_api.ratings_df['user_id'] == user_id]
    
    if user_ratings.empty:
        return jsonify({'error': 'User not found'})
    
    stats = {
        'user_id': user_id,
        'total_ratings': len(user_ratings),
        'average_rating': float(user_ratings['rating'].mean()),
        'rating_distribution': user_ratings['rating'].value_counts().to_dict(),
        'recent_ratings': []
    }
    
    # Get recent ratings with product details
    recent_ratings = user_ratings.head(5)
    for _, rating in recent_ratings.iterrows():
        product_info = rec_api.products_df[rec_api.products_df['product_id'] == rating['product_id']].iloc[0]
        stats['recent_ratings'].append({
            'product_id': rating['product_id'],
            'rating': int(rating['rating']),
            'category': product_info['category'],
            'description': product_info['description']
        })
    
    return jsonify(stats)

@app.route('/api/info')
def api_info():
    """Return basic backend stats for health/info check"""
    if not rec_api.is_trained:
        return jsonify({'status': 'error', 'message': 'System not initialized'})
    return jsonify({
        'status': 'ok',
        'total_users': len(rec_api.ratings_df['user_id'].unique()),
        'total_products': len(rec_api.products_df),
        'total_ratings': len(rec_api.ratings_df)
    })

@app.route('/api/products')
def api_products():
    category = request.args.get('category')
    limit = int(request.args.get('limit', 9))
    file_path = os.path.join(DATA_DIR, DATA_FILE)
    if not os.path.exists(file_path):
        return jsonify({'error': f'Dataset not found'}), 404
    try:
        df = pd.read_csv(file_path)
        df.columns = [col.strip() for col in df.columns]
        filtered = df[df['category'].str.lower().str.contains(category.lower(), na=False)]
        products = []
        for _, row in filtered.head(limit).iterrows():
            product = {
                'name': row.get('name'),
                'price': row.get('price'),
                'rating': row.get('rating'),
                'category': row.get('category'),
                'brand': row.get('brand'),
                'details': row.get('details') or row.get('specs') or row.get('description'),
            }
            products.append(product)
        return jsonify({'products': products})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)