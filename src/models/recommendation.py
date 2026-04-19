"""Recommendation system"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """Content-based and collaborative filtering recommendation system."""
    
    def __init__(self, method='content_based', n_recommendations=5):
        """
        Initialize recommendation engine.
        
        Args:
            method (str): 'content_based' or 'collaborative'
            n_recommendations (int): Number of items to recommend
        """
        self.method = method
        self.n_recommendations = n_recommendations
        self.user_item_matrix = None
        self.item_features = None
        self.similarity_matrix = None
    
    def create_user_item_matrix(self, user_data, n_items=100):
        """
        Create synthetic user-item interaction matrix.
        
        Args:
            user_data (pd.DataFrame): User data
            n_items (int): Number of items in game
        
        Returns:
            np.ndarray: User-item interaction matrix
        """
        n_users = len(user_data)
        
        # Create interaction matrix (ratings 0-5 for items user engaged with)
        matrix = np.random.randint(0, 6, size=(n_users, n_items))
        
        # Make it sparse (most users haven't interacted with most items)
        mask = np.random.random((n_users, n_items)) > 0.7
        matrix[mask] = 0
        
        self.user_item_matrix = matrix
        logger.info(f"User-item matrix created: {matrix.shape}")
        
        return matrix
    
    def create_item_features(self, n_items=100, n_features=10):
        """
        Create item feature vectors.
        
        Args:
            n_items (int): Number of items
            n_features (int): Number of features per item
        
        Returns:
            np.ndarray: Item feature matrix
        """
        # Random item features (difficulty, type, rarity, etc.)
        features = np.random.randn(n_items, n_features)
        
        # Normalize
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        
        self.item_features = features
        logger.info(f"Item features created: {features.shape}")
        
        return features
    
    def recommend_content_based(self, user_id, user_data, liked_items, n_recommendations=None):
        """
        Content-based recommendations.
        
        Args:
            user_id (int): User ID
            user_data (pd.DataFrame): User features
            liked_items (list): Items user has interacted with
            n_recommendations (int): Number of recommendations
        
        Returns:
            list: Recommended item IDs
        """
        if n_recommendations is None:
            n_recommendations = self.n_recommendations
        
        if self.item_features is None:
            self.create_item_features()
        
        # Get features of liked items
        liked_items = [i for i in liked_items if i < len(self.item_features)]
        
        if not liked_items:
            return list(np.random.choice(len(self.item_features), n_recommendations, replace=False))
        
        liked_features = self.item_features[liked_items]
        user_profile = liked_features.mean(axis=0)
        
        # Compute similarity
        similarities = cosine_similarity([user_profile], self.item_features)[0]
        
        # Get top recommendations (excluding already liked items)
        recommendations = []
        for idx in np.argsort(similarities)[::-1]:
            if idx not in liked_items and len(recommendations) < n_recommendations:
                recommendations.append(int(idx))
        
        return recommendations
    
    def recommend_collaborative(self, user_id, n_recommendations=None, min_similarity=0.3):
        """
        Collaborative filtering recommendations.
        
        Args:
            user_id (int): User ID
            n_recommendations (int): Number of recommendations
            min_similarity (float): Minimum similarity threshold
        
        Returns:
            list: Recommended item IDs
        """
        if n_recommendations is None:
            n_recommendations = self.n_recommendations
        
        if self.user_item_matrix is None:
            raise ValueError("User-item matrix not created. Call create_user_item_matrix first.")
        
        # Compute user similarity
        user_similarities = cosine_similarity(self.user_item_matrix)
        
        # Find similar users
        similar_users = np.argsort(user_similarities[user_id])[::-1][1:11]  # Top 10 similar
        
        # Aggregate their ratings
        recommendations_scores = {}
        user_items = set(np.where(self.user_item_matrix[user_id] > 0)[0])
        
        for sim_user in similar_users:
            sim_score = user_similarities[user_id, sim_user]
            if sim_score < min_similarity:
                continue
            
            sim_user_items = np.where(self.user_item_matrix[sim_user] > 0)[0]
            for item in sim_user_items:
                if item not in user_items:
                    recommendations_scores[item] = recommendations_scores.get(item, 0) + sim_score
        
        # Get top N
        recommendations = sorted(recommendations_scores.items(), key=lambda x: x[1], reverse=True)
        recommendations = [item[0] for item in recommendations[:n_recommendations]]
        
        return recommendations
    
    def recommend(self, user_id, user_data, liked_items=None):
        """
        Generate recommendations for user.
        
        Args:
            user_id (int): User ID
            user_data (pd.DataFrame): User features/data
            liked_items (list): Items user has interacted with
        
        Returns:
            dict: Recommendation info
        """
        logger.info(f"Generating {self.method} recommendations for user {user_id}")
        
        if self.method == 'content_based':
            if liked_items is None:
                liked_items = []
            recommendations = self.recommend_content_based(user_id, user_data, liked_items)
        else:  # collaborative
            recommendations = self.recommend_collaborative(user_id)
        
        return {
            'user_id': user_id,
            'recommended_items': recommendations,
            'method': self.method,
            'confidence': 'high' if len(recommendations) >= self.n_recommendations else 'low'
        }
    
    def recommend_batch(self, user_ids, user_data, liked_items_list=None):
        """Generate recommendations for multiple users."""
        recommendations = []
        
        for i, user_id in enumerate(user_ids):
            liked_items = liked_items_list[i] if liked_items_list else None
            rec = self.recommend(user_id, user_data, liked_items)
            recommendations.append(rec)
        
        logger.info(f"Generated recommendations for {len(user_ids)} users")
        return recommendations
