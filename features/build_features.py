"""
Feature engineering module for Spotify Recommendation System.
Builds comprehensive features for user-song pairs.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List
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


class FeatureBuilder:
    """Build features for recommendation model."""
    
    def __init__(self):
        """Initialize feature builder."""
        self.config = get_config()
        logger.info("FeatureBuilder initialized")
    
    def build_song_features(self, audio_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build song-level features.
        
        Args:
            audio_df: Audio features DataFrame
            
        Returns:
            DataFrame with song features
        """
        logger.info("Building song-level features...")
        
        song_features = audio_df.copy()
        
        # 1. Popularity-based features
        song_features['popularity_percentile'] = song_features['popularity'].rank(pct=True)
        song_features['is_popular'] = (song_features['popularity'] > 60).astype(int)
        
        # 2. Audio feature combinations
        song_features['energy_valence'] = song_features['energy'] * song_features['valence']
        song_features['danceability_energy'] = song_features['danceability'] * song_features['energy']
        song_features['acoustic_instrumental'] = song_features['acousticness'] * song_features['instrumentalness']
        
        # 3. Tempo categories
        song_features['tempo_category'] = pd.cut(
            song_features['tempo'],
            bins=[0, 90, 120, 150, 300],
            labels=['slow', 'moderate', 'fast', 'very_fast']
        )
        
        # 4. Duration categories
        song_features['duration_category'] = pd.cut(
            song_features['duration_ms'] / 1000,  # Convert to seconds
            bins=[0, 180, 240, 300, 600],
            labels=['short', 'medium', 'long', 'very_long']
        )
        
        # 5. Musical characteristics
        song_features['is_major'] = song_features['mode']  # 1 = major, 0 = minor
        song_features['is_vocal'] = (song_features['instrumentalness'] < 0.5).astype(int)
        song_features['is_live'] = (song_features['liveness'] > 0.8).astype(int)
        
        logger.info(f"Created {len(song_features.columns)} song features")
        
        return song_features
    
    def build_user_features(self, labels_df: pd.DataFrame, 
                           audio_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build user-level features from interaction history.
        
        Args:
            labels_df: Labels DataFrame with user-song interactions
            audio_df: Audio features DataFrame
            
        Returns:
            DataFrame with user features
        """
        logger.info("Building user-level features...")
        
        # Merge labels with audio features
        user_interactions = labels_df.merge(
            audio_df[['track_id', 'acousticness', 'danceability', 'energy', 
                     'valence', 'tempo', 'popularity']],
            on='track_id',
            how='left'
        )
        
        # Aggregate user preferences
        user_features = user_interactions.groupby('user_id').agg({
            'track_id': 'count',  # Total interactions
            'label': 'mean',  # Positive rate
            'acousticness': 'mean',
            'danceability': 'mean',
            'energy': 'mean',
            'valence': 'mean',
            'tempo': 'mean',
            'popularity': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        user_features.columns = [
            'user_id', 'user_total_interactions', 'user_positive_rate',
            'user_avg_acousticness', 'user_avg_danceability', 'user_avg_energy',
            'user_avg_valence', 'user_avg_tempo', 'user_avg_popularity',
            'user_popularity_std'
        ]
        
        # Fill NaN values
        user_features['user_popularity_std'].fillna(0, inplace=True)
        
        # Additional user features
        user_features['user_is_active'] = (user_features['user_total_interactions'] > 30).astype(int)
        user_features['user_diversity'] = user_features['user_popularity_std'] / (user_features['user_avg_popularity'] + 1)
        
        logger.info(f"Created features for {len(user_features)} users")
        logger.info(f"User features: {list(user_features.columns)}")
        
        return user_features
    
    def build_interaction_features(self, labels_df: pd.DataFrame,
                                   audio_df: pd.DataFrame,
                                   user_features: pd.DataFrame) -> pd.DataFrame:
        """
        Build user-song interaction features.
        
        Args:
            labels_df: Labels DataFrame
            audio_df: Audio features DataFrame
            user_features: User features DataFrame
            
        Returns:
            DataFrame with interaction features
        """
        logger.info("Building user-song interaction features...")
        
        # Merge all data
        features_df = labels_df.copy()
        
        # Add song features
        song_cols = [
            'track_id', 'acousticness', 'danceability', 'energy', 'instrumentalness',
            'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'popularity',
            'duration_ms', 'key', 'mode', 'time_signature'
        ]
        features_df = features_df.merge(audio_df[song_cols], on='track_id', how='left')
        
        # Add user features
        features_df = features_df.merge(user_features, on='user_id', how='left')
        
        # Interaction-specific features
        # 1. User-song affinity scores
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
        
        # 2. Popularity match
        features_df['popularity_match'] = abs(
            features_df['popularity'] - features_df['user_avg_popularity']
        )
        
        # 3. Overall affinity score (lower is better)
        features_df['overall_affinity'] = (
            features_df['affinity_acousticness'] +
            features_df['affinity_danceability'] +
            features_df['affinity_energy'] +
            features_df['affinity_valence']
        ) / 4
        
        logger.info(f"Created {len(features_df.columns)} total features")
        
        return features_df
    
    def select_model_features(self, features_df: pd.DataFrame) -> List[str]:
        """
        Select features for model training.
        
        Args:
            features_df: Full features DataFrame
            
        Returns:
            List of feature column names
        """
        # Audio features
        audio_features = [
            'acousticness', 'danceability', 'energy', 'instrumentalness',
            'liveness', 'loudness', 'speechiness', 'tempo', 'valence',
            'duration_ms', 'key', 'mode', 'time_signature'
        ]
        
        # Song features
        song_features = [
            'popularity', 'popularity_percentile'
        ]
        
        # User features
        user_features = [
            'user_total_interactions', 'user_positive_rate',
            'user_avg_acousticness', 'user_avg_danceability', 'user_avg_energy',
            'user_avg_valence', 'user_avg_tempo', 'user_avg_popularity',
            'user_popularity_std', 'user_is_active', 'user_diversity'
        ]
        
        # Interaction features
        interaction_features = [
            'affinity_acousticness', 'affinity_danceability', 'affinity_energy',
            'affinity_valence', 'popularity_match', 'overall_affinity'
        ]
        
        # Combine all
        all_features = audio_features + song_features + user_features + interaction_features
        
        # Filter to only existing columns
        available_features = [f for f in all_features if f in features_df.columns]
        
        logger.info(f"Selected {len(available_features)} features for modeling")
        
        return available_features
    
    def save_features(self, features_df: pd.DataFrame, feature_cols: List[str]):
        """
        Save feature matrix.
        
        Args:
            features_df: Features DataFrame
            feature_cols: List of feature column names
        """
        logger.info("Saving feature matrix...")
        
        # Save full feature matrix
        output_path = self.config.get_path('paths.feature_matrix')
        features_df.to_csv(output_path, index=False)
        
        logger.info(f"Saved feature matrix: {output_path}")
        logger.info(f"  Shape: {features_df.shape}")
        logger.info(f"  Features: {len(feature_cols)}")
        logger.info(f"  Labels: {features_df['label'].value_counts().to_dict()}")
    
    def run_full_feature_engineering(self, audio_df: pd.DataFrame,
                                     labels_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run complete feature engineering pipeline.
        
        Args:
            audio_df: Audio features DataFrame
            labels_df: Labels DataFrame
            
        Returns:
            Feature matrix DataFrame
        """
        logger.info("=" * 60)
        logger.info("Starting Feature Engineering Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Build song features
        logger.info("\n[Step 1/4] Building song features...")
        song_features = self.build_song_features(audio_df)
        
        # Step 2: Build user features
        logger.info("\n[Step 2/4] Building user features...")
        user_features = self.build_user_features(labels_df, audio_df)
        
        # Step 3: Build interaction features
        logger.info("\n[Step 3/4] Building interaction features...")
        features_df = self.build_interaction_features(labels_df, song_features, user_features)
        
        # Step 4: Select and save features
        logger.info("\n[Step 4/4] Selecting and saving features...")
        feature_cols = self.select_model_features(features_df)
        self.save_features(features_df, feature_cols)
        
        logger.info("\n" + "=" * 60)
        logger.info("Feature Engineering Complete!")
        logger.info("=" * 60)
        
        return features_df, feature_cols


def main():
    """Test feature engineering."""
    # Load data
    logger.info("Loading processed data...")
    config = get_config()
    
    audio_path = config.get_path('paths.audio_features_combined')
    labels_path = config.get_path('paths.labels')
    
    if not audio_path.exists() or not labels_path.exists():
        logger.error("Processed data not found! Run preprocessing and labeling first.")
        return
    
    audio_df = pd.read_csv(audio_path)
    labels_df = pd.read_csv(labels_path)
    
    logger.info(f"Loaded {len(audio_df):,} tracks")
    logger.info(f"Loaded {len(labels_df):,} labeled examples")
    
    # Build features
    builder = FeatureBuilder()
    features_df, feature_cols = builder.run_full_feature_engineering(audio_df, labels_df)
    
    # Display results
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING RESULTS")
    print("=" * 60)
    print(f"\nFeature Matrix Shape: {features_df.shape}")
    print(f"\nSelected Features ({len(feature_cols)}):")
    for i, feat in enumerate(feature_cols, 1):
        print(f"  {i}. {feat}")
    print(f"\nLabel Distribution:")
    print(features_df['label'].value_counts())
    print(f"\nSample features:")
    print(features_df[feature_cols + ['label']].head())


if __name__ == "__main__":
    main()
