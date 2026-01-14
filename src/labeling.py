"""
Labeling module for Spotify Recommendation System.
Generates binary labels for training the recommendation model.

Since the challenge_set playlists are empty (for prediction only),
we'll use a synthetic approach based on audio feature similarity.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, List
from sklearn.metrics.pairwise import cosine_similarity
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


class LabelGenerator:
    """Generate training labels from audio features and simulated user behavior."""
    
    def __init__(self):
        """Initialize label generator."""
        self.config = get_config()
        logger.info("LabelGenerator initialized")
    
    def create_synthetic_users(self, n_users: int = 1000) -> List[int]:
        """
        Create synthetic user IDs.
        
        Args:
            n_users: Number of synthetic users to create
            
        Returns:
            List of user IDs
        """
        logger.info(f"Creating {n_users:,} synthetic users...")
        return list(range(1, n_users + 1))
    
    def generate_user_preferences(self, audio_df: pd.DataFrame, 
                                  n_users: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic user preferences based on audio feature clusters.
        
        Args:
            audio_df: Audio features DataFrame
            n_users: Number of users to simulate
            
        Returns:
            DataFrame with user preferences
        """
        logger.info("Generating user preferences...")
        
        # Select audio features for similarity
        feature_cols = [
            'acousticness', 'danceability', 'energy', 'instrumentalness',
            'liveness', 'loudness', 'speechiness', 'tempo', 'valence'
        ]
        
        # Get feature matrix
        features = audio_df[feature_cols].fillna(0).values
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Create user preference profiles (random points in feature space)
        np.random.seed(42)
        user_profiles = np.random.randn(n_users, len(feature_cols))
        
        logger.info(f"Created {n_users} user preference profiles")
        
        return user_profiles, features_scaled, scaler
    
    def generate_interactions(self, audio_df: pd.DataFrame,
                             n_users: int = 1000,
                             interactions_per_user: int = 50) -> pd.DataFrame:
        """
        Generate synthetic user-song interactions.
        
        Args:
            audio_df: Audio features DataFrame
            n_users: Number of users
            interactions_per_user: Average interactions per user
            
        Returns:
            DataFrame with interactions
        """
        logger.info(f"Generating interactions for {n_users} users...")
        logger.info(f"Target: ~{interactions_per_user} interactions per user")
        
        # Get audio features
        feature_cols = [
            'acousticness', 'danceability', 'energy', 'instrumentalness',
            'liveness', 'loudness', 'speechiness', 'tempo', 'valence'
        ]
        
        # Prepare features
        features = audio_df[feature_cols].fillna(0).values
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Create user profiles
        np.random.seed(42)
        user_profiles = np.random.randn(n_users, len(feature_cols))
        
        interactions = []
        
        logger.info("Computing user-song affinities...")
        
        for user_id in range(n_users):
            # Get user profile
            user_profile = user_profiles[user_id].reshape(1, -1)
            
            # Compute similarity with all songs
            similarities = cosine_similarity(user_profile, features_scaled)[0]
            
            # Add noise to make it realistic
            similarities += np.random.normal(0, 0.1, len(similarities))
            
            # Get top songs for this user
            n_interactions = np.random.poisson(interactions_per_user)
            n_interactions = min(n_interactions, len(audio_df))
            
            # Select songs with probability proportional to similarity
            probabilities = np.exp(similarities * 2)  # Temperature scaling
            probabilities = probabilities / probabilities.sum()
            
            selected_indices = np.random.choice(
                len(audio_df),
                size=n_interactions,
                replace=False,
                p=probabilities
            )
            
            # Create interactions
            for idx in selected_indices:
                interactions.append({
                    'user_id': user_id + 1,
                    'track_id': audio_df.iloc[idx]['track_id'],
                    'track_name': audio_df.iloc[idx]['track_name'],
                    'artist_name': audio_df.iloc[idx]['artist_name'],
                    'similarity_score': similarities[idx]
                })
            
            if (user_id + 1) % 100 == 0:
                logger.info(f"Processed {user_id + 1}/{n_users} users")
        
        df_interactions = pd.DataFrame(interactions)
        logger.info(f"Generated {len(df_interactions):,} total interactions")
        logger.info(f"Average per user: {len(df_interactions) / n_users:.1f}")
        
        return df_interactions
    
    def generate_labels(self, interactions_df: pd.DataFrame,
                       audio_df: pd.DataFrame,
                       replay_threshold: float = 0.6) -> pd.DataFrame:
        """
        Generate binary labels based on interaction patterns.
        
        Label = 1 if:
        - User has high affinity with song (similarity > threshold)
        - Or user interacted with song multiple times
        
        Label = 0 otherwise
        
        Args:
            interactions_df: User-song interactions
            audio_df: Audio features
            replay_threshold: Similarity threshold for positive label
            
        Returns:
            DataFrame with labels
        """
        logger.info("Generating binary labels...")
        
        # Count interactions per user-song pair
        interaction_counts = interactions_df.groupby(['user_id', 'track_id']).size().reset_index(name='play_count')
        
        # Merge with similarity scores
        labels_df = interactions_df.merge(
            interaction_counts,
            on=['user_id', 'track_id'],
            how='left'
        )
        
        # Remove duplicates (keep first interaction)
        labels_df = labels_df.drop_duplicates(subset=['user_id', 'track_id'], keep='first')
        
        # Generate labels
        # Label = 1 if similarity > threshold OR played multiple times
        labels_df['label'] = (
            (labels_df['similarity_score'] > replay_threshold) |
            (labels_df['play_count'] > 1)
        ).astype(int)
        
        # Add negative samples (songs user didn't interact with)
        logger.info("Adding negative samples...")
        
        negative_samples = []
        n_negative_per_user = 30  # Add 30 negative samples per user
        
        all_track_ids = set(audio_df['track_id'].values)
        
        for user_id in labels_df['user_id'].unique():
            # Get songs this user interacted with
            user_tracks = set(labels_df[labels_df['user_id'] == user_id]['track_id'].values)
            
            # Get songs user didn't interact with
            available_tracks = list(all_track_ids - user_tracks)
            
            # Sample negative examples
            if len(available_tracks) > n_negative_per_user:
                sampled_tracks = np.random.choice(
                    available_tracks,
                    size=n_negative_per_user,
                    replace=False
                )
            else:
                sampled_tracks = available_tracks
            
            for track_id in sampled_tracks:
                track_info = audio_df[audio_df['track_id'] == track_id].iloc[0]
                negative_samples.append({
                    'user_id': user_id,
                    'track_id': track_id,
                    'track_name': track_info['track_name'],
                    'artist_name': track_info['artist_name'],
                    'similarity_score': 0.0,
                    'play_count': 0,
                    'label': 0
                })
        
        df_negative = pd.DataFrame(negative_samples)
        labels_df = pd.concat([labels_df, df_negative], ignore_index=True)
        
        # Log statistics
        total = len(labels_df)
        positive = (labels_df['label'] == 1).sum()
        negative = (labels_df['label'] == 0).sum()
        
        logger.info(f"Total labeled examples: {total:,}")
        logger.info(f"  Positive (label=1): {positive:,} ({positive/total*100:.1f}%)")
        logger.info(f"  Negative (label=0): {negative:,} ({negative/total*100:.1f}%)")
        
        return labels_df
    
    def save_labels(self, labels_df: pd.DataFrame):
        """
        Save labeled dataset.
        
        Args:
            labels_df: DataFrame with labels
        """
        logger.info("Saving labeled dataset...")
        
        output_path = self.config.get_path('paths.labels')
        labels_df.to_csv(output_path, index=False)
        
        logger.info(f"Saved labels to: {output_path}")
        logger.info(f"  Shape: {labels_df.shape}")
    
    def run_full_labeling(self, audio_df: pd.DataFrame,
                         n_users: int = 1000,
                         interactions_per_user: int = 50) -> pd.DataFrame:
        """
        Run complete labeling pipeline.
        
        Args:
            audio_df: Audio features DataFrame
            n_users: Number of synthetic users
            interactions_per_user: Interactions per user
            
        Returns:
            Labeled DataFrame
        """
        logger.info("=" * 60)
        logger.info("Starting Label Generation Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Generate interactions
        logger.info("\n[Step 1/3] Generating synthetic interactions...")
        interactions_df = self.generate_interactions(
            audio_df, n_users, interactions_per_user
        )
        
        # Step 2: Generate labels
        logger.info("\n[Step 2/3] Generating binary labels...")
        labels_df = self.generate_labels(interactions_df, audio_df)
        
        # Step 3: Save labels
        logger.info("\n[Step 3/3] Saving labeled dataset...")
        self.save_labels(labels_df)
        
        logger.info("\n" + "=" * 60)
        logger.info("Label Generation Complete!")
        logger.info("=" * 60)
        
        return labels_df


def main():
    """Test label generation."""
    from src.preprocessing import DataPreprocessor
    
    # Load processed audio features
    logger.info("Loading processed audio features...")
    config = get_config()
    audio_path = config.get_path('paths.audio_features_combined')
    
    if not audio_path.exists():
        logger.warning("Processed audio features not found. Running preprocessing first...")
        preprocessor = DataPreprocessor()
        audio_df, _ = preprocessor.run_full_preprocessing(max_playlists=100)
    else:
        audio_df = pd.read_csv(audio_path)
        logger.info(f"Loaded {len(audio_df):,} tracks")
    
    # Generate labels
    generator = LabelGenerator()
    labels_df = generator.run_full_labeling(
        audio_df,
        n_users=500,  # Smaller for testing
        interactions_per_user=40
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("LABEL GENERATION RESULTS")
    print("=" * 60)
    print(f"\nLabeled Dataset Shape: {labels_df.shape}")
    print(f"\nLabel Distribution:")
    print(labels_df['label'].value_counts())
    print(f"\nFirst 5 positive examples:")
    print(labels_df[labels_df['label'] == 1].head())
    print(f"\nFirst 5 negative examples:")
    print(labels_df[labels_df['label'] == 0].head())


if __name__ == "__main__":
    main()

