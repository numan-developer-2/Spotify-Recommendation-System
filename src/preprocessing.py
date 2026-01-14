"""
Preprocessing module for Spotify Recommendation System.
Cleans and prepares raw data for feature engineering and modeling.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import get_config
from src.data_loader import DataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess and clean datasets."""
    
    def __init__(self):
        """Initialize preprocessor with configuration."""
        self.config = get_config()
        self.loader = DataLoader()
        logger.info("DataPreprocessor initialized")
    
    def clean_audio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean audio features dataset.
        
        Args:
            df: Raw audio features DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning audio features dataset...")
        logger.info(f"Initial shape: {df.shape}")
        
        df_clean = df.copy()
        
        # 1. Handle missing values
        logger.info("Handling missing values...")
        missing_before = df_clean.isnull().sum().sum()
        
        # Fill missing track names with placeholder
        if df_clean['track_name'].isnull().any():
            df_clean['track_name'].fillna('Unknown Track', inplace=True)
            logger.info(f"Filled {df_clean['track_name'].isnull().sum()} missing track names")
        
        # Fill missing artist names
        if df_clean['artist_name'].isnull().any():
            df_clean['artist_name'].fillna('Unknown Artist', inplace=True)
        
        # Fill numerical features with median
        numerical_cols = [
            'acousticness', 'danceability', 'energy', 'instrumentalness',
            'liveness', 'loudness', 'speechiness', 'tempo', 'valence'
        ]
        
        for col in numerical_cols:
            if df_clean[col].isnull().any():
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                logger.info(f"Filled missing {col} with median: {median_val:.4f}")
        
        missing_after = df_clean.isnull().sum().sum()
        logger.info(f"Missing values: {missing_before} → {missing_after}")
        
        # 2. Remove duplicates based on track_id
        duplicates = df_clean.duplicated(subset=['track_id'], keep='first').sum()
        if duplicates > 0:
            df_clean = df_clean.drop_duplicates(subset=['track_id'], keep='first')
            logger.info(f"Removed {duplicates} duplicate tracks")
        
        # 3. Validate numerical ranges
        logger.info("Validating numerical ranges...")
        
        # Clip values to valid ranges (0-1 for most features)
        for col in ['acousticness', 'danceability', 'energy', 'instrumentalness',
                    'liveness', 'speechiness', 'valence']:
            df_clean[col] = df_clean[col].clip(0, 1)
        
        # Tempo should be positive
        df_clean['tempo'] = df_clean['tempo'].clip(lower=0)
        
        # Duration should be positive
        df_clean['duration_ms'] = df_clean['duration_ms'].clip(lower=0)
        
        # Popularity should be 0-100
        df_clean['popularity'] = df_clean['popularity'].clip(0, 100)
        
        # 4. Add derived features
        logger.info("Adding derived features...")
        
        # Duration in minutes
        df_clean['duration_min'] = df_clean['duration_ms'] / 60000
        
        # Categorize popularity
        df_clean['popularity_category'] = pd.cut(
            df_clean['popularity'],
            bins=[0, 20, 40, 60, 80, 100],
            labels=['very_low', 'low', 'medium', 'high', 'very_high']
        )
        
        # Energy level category
        df_clean['energy_level'] = pd.cut(
            df_clean['energy'],
            bins=[0, 0.3, 0.6, 1.0],
            labels=['low', 'medium', 'high']
        )
        
        logger.info(f"Final shape: {df_clean.shape}")
        logger.info(f"Cleaned dataset: {len(df_clean):,} tracks")
        
        return df_clean
    
    def process_playlists(self, playlist_data: Dict) -> pd.DataFrame:
        """
        Convert playlist JSON to structured DataFrame.
        
        Args:
            playlist_data: Playlist dictionary from DataLoader
            
        Returns:
            DataFrame with playlist-track interactions
        """
        logger.info("Processing playlist data...")
        
        playlists = playlist_data['playlists']
        logger.info(f"Processing {len(playlists):,} playlists")
        
        interactions = []
        
        for playlist in playlists:
            pid = playlist['pid']
            playlist_name = playlist.get('name', None)
            tracks = playlist.get('tracks', [])
            
            for track in tracks:
                interactions.append({
                    'user_id': pid,  # Playlist ID as user ID
                    'playlist_name': playlist_name,
                    'track_uri': track.get('track_uri'),
                    'track_name': track.get('track_name'),
                    'artist_name': track.get('artist_name'),
                    'artist_uri': track.get('artist_uri'),
                    'album_name': track.get('album_name'),
                    'album_uri': track.get('album_uri'),
                    'duration_ms': track.get('duration_ms'),
                    'position': track.get('pos')  # Position in playlist
                })
        
        df_interactions = pd.DataFrame(interactions)
        
        if len(df_interactions) == 0:
            logger.warning("No interactions found! Playlists may be empty (challenge set).")
            logger.warning("Creating empty DataFrame with expected schema...")
            df_interactions = pd.DataFrame(columns=[
                'user_id', 'playlist_name', 'track_uri', 'track_name',
                'artist_name', 'artist_uri', 'album_name', 'album_uri',
                'duration_ms', 'position'
            ])
            return df_interactions
        
        logger.info(f"Created {len(df_interactions):,} interactions")
        logger.info(f"Unique users (playlists): {df_interactions['user_id'].nunique():,}")
        logger.info(f"Unique tracks: {df_interactions['track_uri'].nunique():,}")
        
        return df_interactions
    
    def normalize_features(self, df: pd.DataFrame, 
                          features: List[str] = None) -> pd.DataFrame:
        """
        Normalize numerical features to 0-1 range.
        
        Args:
            df: DataFrame with features
            features: List of feature names to normalize (None = auto-detect)
            
        Returns:
            DataFrame with normalized features
        """
        logger.info("Normalizing features...")
        
        df_norm = df.copy()
        
        if features is None:
            # Auto-detect numerical features
            features = [
                'acousticness', 'danceability', 'energy', 'instrumentalness',
                'liveness', 'loudness', 'speechiness', 'tempo', 'valence',
                'duration_ms', 'popularity'
            ]
            # Only keep features that exist in DataFrame
            features = [f for f in features if f in df_norm.columns]
        
        for feature in features:
            min_val = df_norm[feature].min()
            max_val = df_norm[feature].max()
            
            if max_val > min_val:
                df_norm[f'{feature}_norm'] = (df_norm[feature] - min_val) / (max_val - min_val)
                logger.info(f"Normalized {feature}: [{min_val:.2f}, {max_val:.2f}] → [0, 1]")
            else:
                df_norm[f'{feature}_norm'] = 0
                logger.warning(f"Feature {feature} has constant value: {min_val}")
        
        return df_norm
    
    def save_processed_data(self, audio_df: pd.DataFrame, 
                           interactions_df: pd.DataFrame):
        """
        Save processed datasets to disk.
        
        Args:
            audio_df: Processed audio features
            interactions_df: Processed interactions
        """
        logger.info("Saving processed datasets...")
        
        # Create processed directory if needed
        processed_dir = self.config.get_path('paths.processed_data_dir', create_dir=True)
        
        # Save audio features
        audio_path = self.config.get_path('paths.audio_features_combined')
        audio_df.to_csv(audio_path, index=False)
        logger.info(f"Saved audio features: {audio_path}")
        logger.info(f"  Shape: {audio_df.shape}")
        
        # Save interactions
        interactions_path = self.config.get_path('paths.interactions')
        interactions_df.to_csv(interactions_path, index=False)
        logger.info(f"Saved interactions: {interactions_path}")
        logger.info(f"  Shape: {interactions_df.shape}")
        
        logger.info("All processed data saved successfully!")
    
    def run_full_preprocessing(self, max_playlists: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run complete preprocessing pipeline.
        
        Args:
            max_playlists: Maximum playlists to process (None = all)
            
        Returns:
            Tuple of (audio_features_df, interactions_df)
        """
        logger.info("=" * 60)
        logger.info("Starting Full Preprocessing Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Load raw data
        logger.info("\n[Step 1/4] Loading raw datasets...")
        audio_raw, playlist_raw = self.loader.load_all_datasets(max_playlists)
        
        # Step 2: Clean audio features
        logger.info("\n[Step 2/4] Cleaning audio features...")
        audio_clean = self.clean_audio_features(audio_raw)
        
        # Step 3: Process playlists
        logger.info("\n[Step 3/4] Processing playlists...")
        interactions = self.process_playlists(playlist_raw)
        
        # Step 4: Save processed data
        logger.info("\n[Step 4/4] Saving processed data...")
        self.save_processed_data(audio_clean, interactions)
        
        logger.info("\n" + "=" * 60)
        logger.info("Preprocessing Complete!")
        logger.info("=" * 60)
        
        return audio_clean, interactions


def main():
    """Test preprocessing functionality."""
    preprocessor = DataPreprocessor()
    
    # Run full preprocessing with limited playlists for testing
    logger.info("Running preprocessing with 1000 playlists for testing...")
    audio_df, interactions_df = preprocessor.run_full_preprocessing(max_playlists=1000)
    
    # Display results
    print("\n" + "=" * 60)
    print("PREPROCESSING RESULTS")
    print("=" * 60)
    
    print(f"\nAudio Features Dataset:")
    print(f"  Shape: {audio_df.shape}")
    print(f"  Columns: {list(audio_df.columns)}")
    print(f"\nFirst 3 rows:")
    print(audio_df.head(3))
    
    print(f"\n\nInteractions Dataset:")
    print(f"  Shape: {interactions_df.shape}")
    print(f"  Unique users: {interactions_df['user_id'].nunique()}")
    print(f"  Unique tracks: {interactions_df['track_uri'].nunique()}")
    print(f"\nFirst 3 rows:")
    print(interactions_df.head(3))


if __name__ == "__main__":
    main()

