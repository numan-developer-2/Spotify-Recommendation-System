"""
Recommendation module for Spotify Recommendation System.
Generates personalized music recommendations.
"""

import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import logging
from pathlib import Path
from typing import List, Dict
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import get_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RecommendationEngine:
    """Generate personalized music recommendations."""
    
    def __init__(self, model_type: str = 'lightgbm'):
        """
        Initialize recommendation engine.
        
        Args:
            model_type: Type of model to use ('lightgbm' or 'baseline')
        """
        self.config = get_config()
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.audio_df = None
        self.user_features = None
        
        self._load_model()
        self._load_data()
        
        logger.info(f"RecommendationEngine initialized with {model_type} model")
    
    def _load_model(self):
        """Load trained model."""
        logger.info(f"Loading {self.model_type} model...")
        
        models_dir = self.config.get_path('paths.models_dir')
        
        if self.model_type == 'lightgbm':
            model_path = models_dir / 'lightgbm_model.txt'
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            self.model = lgb.Booster(model_file=str(model_path))
            logger.info(f"Loaded LightGBM model from: {model_path}")
            
        elif self.model_type == 'baseline':
            model_path = models_dir / 'baseline_lr.pkl'
            scaler_path = models_dir / 'baseline_scaler.pkl'
            
            if not model_path.exists() or not scaler_path.exists():
                raise FileNotFoundError(f"Model files not found")
            
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Loaded baseline model from: {model_path}")
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _load_data(self):
        """Load audio features and feature matrix."""
        logger.info("Loading data...")
        
        # Load audio features
        audio_path = self.config.get_path('paths.audio_features_combined')
        self.audio_df = pd.read_csv(audio_path)
        logger.info(f"Loaded {len(self.audio_df):,} tracks")
        
        # Load feature matrix to get feature columns and user features
        feature_path = self.config.get_path('paths.feature_matrix')
        feature_df = pd.read_csv(feature_path)
        
        # Get feature columns
        exclude_cols = [
            'user_id', 'track_id', 'track_name', 'artist_name', 
            'label', 'similarity_score', 'play_count',
            'playlist_name', 'track_uri', 'artist_uri', 'album_name', 'album_uri',
            'duration_category', 'tempo_category', 'popularity_category', 'energy_level',
            'data_source'
        ]
        self.feature_cols = [col for col in feature_df.columns if col not in exclude_cols]
        
        # Extract user features
        user_feature_cols = [col for col in self.feature_cols if col.startswith('user_')]
        self.user_features = feature_df[['user_id'] + user_feature_cols].drop_duplicates('user_id')
        
        logger.info(f"Loaded {len(self.feature_cols)} features")
        logger.info(f"Loaded {len(self.user_features)} user profiles")
    
    def get_candidate_songs(self, user_id: int, exclude_tracks: List[str] = None,
                           max_candidates: int = 1000) -> pd.DataFrame:
        """
        Get candidate songs for recommendation.
        
        Args:
            user_id: User ID
            exclude_tracks: List of track IDs to exclude (already listened)
            max_candidates: Maximum number of candidates
            
        Returns:
            DataFrame with candidate songs
        """
        candidates = self.audio_df.copy()
        
        # Exclude already listened tracks
        if exclude_tracks:
            candidates = candidates[~candidates['track_id'].isin(exclude_tracks)]
        
        # Filter by minimum popularity
        min_popularity = self.config.get('recommendation.min_popularity', 10)
        candidates = candidates[candidates['popularity'] >= min_popularity]
        
        # Limit candidates
        if len(candidates) > max_candidates:
            # Sample diverse candidates
            candidates = candidates.sample(n=max_candidates, random_state=42)
        
        logger.info(f"Generated {len(candidates)} candidate songs for user {user_id}")
        
        return candidates
    
    def build_features_for_candidates(self, user_id: int, 
                                     candidates_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build features for user-candidate pairs.
        
        Args:
            user_id: User ID
            candidates_df: Candidate songs DataFrame
            
        Returns:
            DataFrame with features
        """
        # Get user features
        user_feat = self.user_features[self.user_features['user_id'] == user_id]
        
        if len(user_feat) == 0:
            logger.warning(f"User {user_id} not found. Using average user profile.")
            # Create average user profile
            user_feat = pd.DataFrame([{
                'user_id': user_id,
                **{col: self.user_features[col].mean() 
                   for col in self.user_features.columns if col != 'user_id'}
            }])
        
        # Merge candidates with user features
        features_df = candidates_df.copy()
        features_df['user_id'] = user_id
        features_df = features_df.merge(user_feat, on='user_id', how='left')
        
        # Build interaction features
        features_df['affinity_acousticness'] = abs(
            features_df['acousticness'] - features_df['user_avg_acousticness']
        )
        features_df['affinity_danceability'] = abs(
            features_df['danceability'] - features_df['user_avg_danceability']
        )
        features_df['affinity_energy'] = abs(
            features_df['energy'] - features_df['user_avg_energy']
        )
        features_df['affinity_valence'] = abs(
            features_df['valence'] - features_df['user_avg_valence']
        )
        features_df['popularity_match'] = abs(
            features_df['popularity'] - features_df['user_avg_popularity']
        )
        features_df['overall_affinity'] = (
            features_df['affinity_acousticness'] +
            features_df['affinity_danceability'] +
            features_df['affinity_energy'] +
            features_df['affinity_valence']
        ) / 4
        
        # Add missing features if needed
        for col in self.feature_cols:
            if col not in features_df.columns:
                features_df[col] = 0
        
        return features_df
    
    def score_candidates(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Score candidates using trained model.
        
        Args:
            features_df: Features DataFrame
            
        Returns:
            Array of scores
        """
        # Select features in correct order
        X = features_df[self.feature_cols].fillna(0)
        
        # Predict
        if self.model_type == 'lightgbm':
            scores = self.model.predict(X)
        else:  # baseline
            X_scaled = self.scaler.transform(X)
            scores = self.model.predict_proba(X_scaled)[:, 1]
        
        return scores
    
    def rank_and_filter(self, candidates_df: pd.DataFrame, scores: np.ndarray,
                       top_k: int = 10) -> pd.DataFrame:
        """
        Rank candidates and apply diversity filters.
        
        Args:
            candidates_df: Candidates DataFrame
            scores: Prediction scores
            top_k: Number of recommendations to return
            
        Returns:
            Top-K recommendations DataFrame
        """
        # Add scores
        candidates_df = candidates_df.copy()
        candidates_df['score'] = scores
        
        # Sort by score
        candidates_df = candidates_df.sort_values('score', ascending=False)
        
        # Apply diversity: ensure variety in artists
        recommendations = []
        seen_artists = set()
        
        for idx, row in candidates_df.iterrows():
            artist = row['artist_name']
            
            # Allow max 2 songs per artist in top-K
            artist_count = sum(1 for r in recommendations if r['artist_name'] == artist)
            
            if artist_count < 2:
                recommendations.append(row)
                seen_artists.add(artist)
            
            if len(recommendations) >= top_k:
                break
        
        # If not enough diverse recommendations, fill with remaining top scores
        if len(recommendations) < top_k:
            remaining = candidates_df[~candidates_df.index.isin([r.name for r in recommendations])]
            recommendations.extend(remaining.head(top_k - len(recommendations)).to_dict('records'))
        
        recommendations_df = pd.DataFrame(recommendations)
        
        return recommendations_df
    
    def recommend(self, user_id: int, exclude_tracks: List[str] = None,
                 top_k: int = 10) -> pd.DataFrame:
        """
        Generate recommendations for a user.
        
        Args:
            user_id: User ID
            exclude_tracks: Tracks to exclude
            top_k: Number of recommendations
            
        Returns:
            DataFrame with recommendations
        """
        logger.info(f"Generating {top_k} recommendations for user {user_id}...")
        
        # Step 1: Get candidates
        candidates = self.get_candidate_songs(user_id, exclude_tracks)
        
        # Step 2: Build features
        features_df = self.build_features_for_candidates(user_id, candidates)
        
        # Step 3: Score candidates
        scores = self.score_candidates(features_df)
        
        # Step 4: Rank and filter
        recommendations = self.rank_and_filter(features_df, scores, top_k)
        
        # Select output columns
        output_cols = [
            'track_name', 'artist_name', 'popularity', 'score',
            'danceability', 'energy', 'valence', 'tempo'
        ]
        recommendations = recommendations[output_cols]
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        
        return recommendations


def main():
    """Test recommendation engine."""
    # Initialize engine
    engine = RecommendationEngine(model_type='lightgbm')
    
    # Test recommendations for a few users
    test_users = [1, 10, 50, 100]
    
    for user_id in test_users:
        print("\n" + "=" * 60)
        print(f"Recommendations for User {user_id}")
        print("=" * 60)
        
        recommendations = engine.recommend(user_id, top_k=10)
        
        print(f"\nTop 10 Recommended Songs:")
        for idx, row in recommendations.iterrows():
            print(f"\n{idx + 1}. {row['track_name']}")
            print(f"   Artist: {row['artist_name']}")
            print(f"   Score: {row['score']:.4f} | Popularity: {row['popularity']}")
            print(f"   Danceability: {row['danceability']:.2f} | Energy: {row['energy']:.2f}")


if __name__ == "__main__":
    main()

