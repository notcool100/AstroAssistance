"""
Feature engineering for AstroAssistance.

This module provides advanced feature engineering capabilities for the models.
"""
import os
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
import torch
from torch.utils.data import Dataset, DataLoader

from src.core.config import config_manager
from src.core.logger import app_logger
from src.core.data_types import (
    Task, TaskPriority, TaskStatus, TaskCategory,
    Reminder, Goal, UserPreference, UserActivity,
    LearningFeedback, Recommendation, RecurrencePattern
)


class FeatureEngineer:
    """Advanced feature engineering for model training."""
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.processed_data_dir = os.path.join(self.project_root, "data", "processed")
        
        # Initialize transformers
        self.transformers = {}
    
    def extract_temporal_features(self, df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
        """
        Extract temporal features from a datetime column.
        
        Args:
            df: Input DataFrame
            datetime_col: Name of the datetime column
            
        Returns:
            DataFrame with additional temporal features
        """
        if datetime_col not in df.columns:
            app_logger.warning(f"Column {datetime_col} not found in DataFrame")
            return df
        
        # Ensure column is datetime type
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
        
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Extract basic temporal features
        result_df[f"{datetime_col}_hour"] = df[datetime_col].dt.hour
        result_df[f"{datetime_col}_day"] = df[datetime_col].dt.day
        result_df[f"{datetime_col}_day_of_week"] = df[datetime_col].dt.dayofweek
        result_df[f"{datetime_col}_day_of_year"] = df[datetime_col].dt.dayofyear
        result_df[f"{datetime_col}_month"] = df[datetime_col].dt.month
        result_df[f"{datetime_col}_quarter"] = df[datetime_col].dt.quarter
        result_df[f"{datetime_col}_year"] = df[datetime_col].dt.year
        
        # Extract cyclical features for hour, day of week, month
        # This preserves the cyclical nature of these features
        result_df[f"{datetime_col}_hour_sin"] = np.sin(2 * np.pi * df[datetime_col].dt.hour / 24)
        result_df[f"{datetime_col}_hour_cos"] = np.cos(2 * np.pi * df[datetime_col].dt.hour / 24)
        
        result_df[f"{datetime_col}_day_of_week_sin"] = np.sin(2 * np.pi * df[datetime_col].dt.dayofweek / 7)
        result_df[f"{datetime_col}_day_of_week_cos"] = np.cos(2 * np.pi * df[datetime_col].dt.dayofweek / 7)
        
        result_df[f"{datetime_col}_month_sin"] = np.sin(2 * np.pi * df[datetime_col].dt.month / 12)
        result_df[f"{datetime_col}_month_cos"] = np.cos(2 * np.pi * df[datetime_col].dt.month / 12)
        
        # Is weekend
        result_df[f"{datetime_col}_is_weekend"] = (df[datetime_col].dt.dayofweek >= 5).astype(int)
        
        # Is business hours (9-17)
        result_df[f"{datetime_col}_is_business_hours"] = ((df[datetime_col].dt.hour >= 9) & 
                                                         (df[datetime_col].dt.hour < 17)).astype(int)
        
        return result_df
    
    def extract_text_features(self, df: pd.DataFrame, text_col: str, max_features: int = 100) -> pd.DataFrame:
        """
        Extract features from text data.
        
        Args:
            df: Input DataFrame
            text_col: Name of the text column
            max_features: Maximum number of features to extract
            
        Returns:
            DataFrame with additional text features
        """
        if text_col not in df.columns:
            app_logger.warning(f"Column {text_col} not found in DataFrame")
            return df
        
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Basic text features
        result_df[f"{text_col}_length"] = df[text_col].fillna("").apply(len)
        result_df[f"{text_col}_word_count"] = df[text_col].fillna("").apply(lambda x: len(x.split()))
        
        # TF-IDF features
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Fill NaN values
        text_data = df[text_col].fillna("")
        
        # Create and fit vectorizer
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            min_df=2,  # Minimum document frequency
            max_df=0.9  # Maximum document frequency
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(text_data)
            
            # Convert to DataFrame
            tfidf_df = pd.DataFrame(
                tfidf_matrix.toarray(),
                columns=[f"{text_col}_tfidf_{i}" for i in range(tfidf_matrix.shape[1])],
                index=df.index
            )
            
            # Save vectorizer
            self.transformers[f"{text_col}_tfidf"] = vectorizer
            
            # Join with result DataFrame
            result_df = pd.concat([result_df, tfidf_df], axis=1)
        except Exception as e:
            app_logger.error(f"Error extracting TF-IDF features: {str(e)}")
        
        return result_df
    
    def create_interaction_features(self, df: pd.DataFrame, feature_cols: List[str], 
                                   degree: int = 2, interaction_only: bool = True) -> pd.DataFrame:
        """
        Create interaction features between numerical columns.
        
        Args:
            df: Input DataFrame
            feature_cols: List of feature columns to use
            degree: Degree of polynomial features
            interaction_only: Whether to include only interaction terms
            
        Returns:
            DataFrame with additional interaction features
        """
        # Filter to only include columns that exist
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        if not feature_cols:
            app_logger.warning("No valid feature columns provided")
            return df
        
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Select only numerical columns
        num_df = df[feature_cols].select_dtypes(include=["int64", "float64"])
        
        if num_df.empty:
            app_logger.warning("No numerical columns found in feature_cols")
            return df
        
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
        
        try:
            # Fill NaN values
            num_df_filled = num_df.fillna(0)
            
            # Transform data
            poly_features = poly.fit_transform(num_df_filled)
            
            # Get feature names
            feature_names = poly.get_feature_names_out(num_df.columns)
            
            # Remove original features (they're already in the DataFrame)
            mask = np.ones(len(feature_names), dtype=bool)
            for i, name in enumerate(feature_names):
                if name in num_df.columns:
                    mask[i] = False
            
            # Filter features and names
            poly_features = poly_features[:, mask]
            feature_names = feature_names[mask]
            
            # Create DataFrame with interaction features
            poly_df = pd.DataFrame(
                poly_features,
                columns=feature_names,
                index=df.index
            )
            
            # Save transformer
            self.transformers["polynomial"] = poly
            
            # Join with result DataFrame
            result_df = pd.concat([result_df, poly_df], axis=1)
        except Exception as e:
            app_logger.error(f"Error creating interaction features: {str(e)}")
        
        return result_df
    
    def create_lag_features(self, df: pd.DataFrame, time_col: str, feature_cols: List[str], 
                           group_col: str, lag_periods: List[int]) -> pd.DataFrame:
        """
        Create lag features for time series data.
        
        Args:
            df: Input DataFrame
            time_col: Name of the time column
            feature_cols: List of feature columns to create lags for
            group_col: Column to group by (e.g., user_id)
            lag_periods: List of lag periods to create
            
        Returns:
            DataFrame with additional lag features
        """
        # Check if required columns exist
        required_cols = [time_col, group_col] + feature_cols
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            app_logger.warning(f"Missing columns: {missing_cols}")
            return df
        
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Ensure time column is datetime type
        result_df[time_col] = pd.to_datetime(result_df[time_col], errors="coerce")
        
        # Sort by group and time
        result_df = result_df.sort_values([group_col, time_col])
        
        # Create lag features
        for feature in feature_cols:
            for lag in lag_periods:
                # Create lag feature
                result_df[f"{feature}_lag_{lag}"] = result_df.groupby(group_col)[feature].shift(lag)
        
        return result_df
    
    def create_rolling_features(self, df: pd.DataFrame, time_col: str, feature_cols: List[str],
                               group_col: str, windows: List[int]) -> pd.DataFrame:
        """
        Create rolling window features for time series data.
        
        Args:
            df: Input DataFrame
            time_col: Name of the time column
            feature_cols: List of feature columns to create rolling features for
            group_col: Column to group by (e.g., user_id)
            windows: List of window sizes to create
            
        Returns:
            DataFrame with additional rolling features
        """
        # Check if required columns exist
        required_cols = [time_col, group_col] + feature_cols
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            app_logger.warning(f"Missing columns: {missing_cols}")
            return df
        
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Ensure time column is datetime type
        result_df[time_col] = pd.to_datetime(result_df[time_col], errors="coerce")
        
        # Sort by group and time
        result_df = result_df.sort_values([group_col, time_col])
        
        # Create rolling features
        for feature in feature_cols:
            for window in windows:
                # Create rolling mean
                result_df[f"{feature}_rolling_mean_{window}"] = (
                    result_df.groupby(group_col)[feature]
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )
                
                # Create rolling std
                result_df[f"{feature}_rolling_std_{window}"] = (
                    result_df.groupby(group_col)[feature]
                    .rolling(window=window, min_periods=1)
                    .std()
                    .reset_index(level=0, drop=True)
                )
                
                # Create rolling min
                result_df[f"{feature}_rolling_min_{window}"] = (
                    result_df.groupby(group_col)[feature]
                    .rolling(window=window, min_periods=1)
                    .min()
                    .reset_index(level=0, drop=True)
                )
                
                # Create rolling max
                result_df[f"{feature}_rolling_max_{window}"] = (
                    result_df.groupby(group_col)[feature]
                    .rolling(window=window, min_periods=1)
                    .max()
                    .reset_index(level=0, drop=True)
                )
        
        return result_df
    
    def create_user_activity_features(self, tasks_df: pd.DataFrame, activities_df: pd.DataFrame, 
                                     user_col: str = "user_id") -> pd.DataFrame:
        """
        Create user activity features by aggregating activity data.
        
        Args:
            tasks_df: Tasks DataFrame
            activities_df: User activities DataFrame
            user_col: Name of the user ID column
            
        Returns:
            DataFrame with additional user activity features
        """
        # Check if required columns exist
        if user_col not in tasks_df.columns or user_col not in activities_df.columns:
            app_logger.warning(f"User column {user_col} not found in one of the DataFrames")
            return tasks_df
        
        # Create a copy to avoid modifying the original
        result_df = tasks_df.copy()
        
        # Aggregate activity data by user
        user_activity_stats = activities_df.groupby(user_col).agg({
            "id": "count",  # Total activities
            "timestamp": ["min", "max"]  # First and last activity
        })
        
        # Flatten multi-level columns
        user_activity_stats.columns = ["_".join(col).strip() for col in user_activity_stats.columns.values]
        
        # Rename columns
        user_activity_stats = user_activity_stats.rename(columns={
            "id_count": "total_activities",
            "timestamp_min": "first_activity",
            "timestamp_max": "last_activity"
        })
        
        # Calculate activity duration in days
        user_activity_stats["activity_duration_days"] = (
            (pd.to_datetime(user_activity_stats["last_activity"]) - 
             pd.to_datetime(user_activity_stats["first_activity"]))
            .dt.total_seconds() / (24 * 3600)
        )
        
        # Calculate activity frequency (activities per day)
        user_activity_stats["activity_frequency"] = (
            user_activity_stats["total_activities"] / 
            user_activity_stats["activity_duration_days"].replace(0, 1)  # Avoid division by zero
        )
        
        # Count activity types if activity_type column exists
        if "activity_type" in activities_df.columns:
            # Get activity type counts
            activity_type_counts = activities_df.groupby([user_col, "activity_type"]).size().unstack(fill_value=0)
            
            # Rename columns
            activity_type_counts.columns = [f"activity_count_{col}" for col in activity_type_counts.columns]
            
            # Join with user activity stats
            user_activity_stats = user_activity_stats.join(activity_type_counts)
        
        # Merge with tasks DataFrame
        result_df = result_df.merge(user_activity_stats, on=user_col, how="left")
        
        return result_df
    
    def create_user_preference_features(self, df: pd.DataFrame, preferences_df: pd.DataFrame,
                                       user_col: str = "user_id") -> pd.DataFrame:
        """
        Create features based on user preferences.
        
        Args:
            df: Input DataFrame
            preferences_df: User preferences DataFrame
            user_col: Name of the user ID column
            
        Returns:
            DataFrame with additional user preference features
        """
        # Check if required columns exist
        if user_col not in df.columns or user_col not in preferences_df.columns:
            app_logger.warning(f"User column {user_col} not found in one of the DataFrames")
            return df
        
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Extract relevant preference features
        pref_features = preferences_df[[user_col]].copy()
        
        # Add productivity peak hours feature
        if "productivity_peak_hours" in preferences_df.columns:
            pref_features["peak_hours_count"] = preferences_df["productivity_peak_hours"].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
        
        # Add work days feature
        if "work_days" in preferences_df.columns:
            pref_features["work_days_count"] = preferences_df["work_days"].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
        
        # Add work hours duration
        if "work_hours" in preferences_df.columns:
            pref_features["work_hours_duration"] = preferences_df["work_hours"].apply(
                lambda x: x.get("end", 17) - x.get("start", 9) if isinstance(x, dict) else 8
            )
        
        # Add preferred task duration
        if "preferred_task_duration" in preferences_df.columns:
            pref_features["preferred_task_duration"] = preferences_df["preferred_task_duration"]
        
        # Add break duration
        if "break_duration" in preferences_df.columns:
            pref_features["break_duration"] = preferences_df["break_duration"]
        
        # Merge with input DataFrame
        result_df = result_df.merge(pref_features, on=user_col, how="left")
        
        return result_df
    
    def select_features(self, X: pd.DataFrame, y: np.ndarray, k: int = 10, method: str = "f_classif") -> pd.DataFrame:
        """
        Select top k features based on statistical tests.
        
        Args:
            X: Feature DataFrame
            y: Target array
            k: Number of features to select
            method: Feature selection method ('f_classif' or 'mutual_info')
            
        Returns:
            DataFrame with selected features
        """
        # Ensure k is not larger than the number of features
        k = min(k, X.shape[1])
        
        # Choose selection method
        if method == "mutual_info":
            selector = SelectKBest(mutual_info_classif, k=k)
        else:  # default to f_classif
            selector = SelectKBest(f_classif, k=k)
        
        # Fit and transform
        X_new = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()]
        
        # Create DataFrame with selected features
        X_selected = pd.DataFrame(X_new, columns=selected_features, index=X.index)
        
        # Save selector
        self.transformers[f"feature_selector_{method}"] = selector
        
        return X_selected
    
    def reduce_dimensions(self, X: pd.DataFrame, n_components: int = 10, method: str = "pca") -> pd.DataFrame:
        """
        Reduce dimensionality of features.
        
        Args:
            X: Feature DataFrame
            n_components: Number of components to keep
            method: Dimensionality reduction method ('pca' only for now)
            
        Returns:
            DataFrame with reduced dimensions
        """
        # Ensure n_components is not larger than the number of features
        n_components = min(n_components, X.shape[1])
        
        # Choose reduction method
        if method == "pca":
            reducer = PCA(n_components=n_components)
        else:
            app_logger.warning(f"Unknown method: {method}, using PCA")
            reducer = PCA(n_components=n_components)
        
        # Fit and transform
        X_reduced = reducer.fit_transform(X)
        
        # Create DataFrame with reduced dimensions
        X_reduced_df = pd.DataFrame(
            X_reduced,
            columns=[f"{method}_{i}" for i in range(n_components)],
            index=X.index
        )
        
        # Save reducer
        self.transformers[f"dimension_reducer_{method}"] = reducer
        
        return X_reduced_df
    
    def save_transformers(self) -> None:
        """Save all transformers."""
        import pickle
        
        # Create directory if it doesn't exist
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        # Save transformers
        for name, transformer in self.transformers.items():
            file_path = os.path.join(self.processed_data_dir, f"{name}_transformer.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(transformer, f)
            
            app_logger.info(f"Saved transformer {name} to {file_path}")
    
    def load_transformers(self) -> None:
        """Load all transformers."""
        import pickle
        
        # Get all transformer files
        transformer_files = [f for f in os.listdir(self.processed_data_dir) if f.endswith("_transformer.pkl")]
        
        for file in transformer_files:
            try:
                file_path = os.path.join(self.processed_data_dir, file)
                with open(file_path, "rb") as f:
                    transformer = pickle.load(f)
                
                # Extract name from filename
                name = file.replace("_transformer.pkl", "")
                self.transformers[name] = transformer
                
                app_logger.info(f"Loaded transformer {name} from {file_path}")
            except Exception as e:
                app_logger.error(f"Error loading transformer {file}: {str(e)}")
    
    def create_all_features(self, tasks_df: pd.DataFrame, activities_df: pd.DataFrame, 
                           recommendations_df: pd.DataFrame, preferences_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Create all features for model training.
        
        Args:
            tasks_df: Tasks DataFrame
            activities_df: User activities DataFrame
            recommendations_df: Recommendations DataFrame
            preferences_df: User preferences DataFrame
            
        Returns:
            Dictionary of DataFrames with enhanced features
        """
        app_logger.info("Creating all features")
        
        # Enhanced tasks features
        tasks_enhanced = tasks_df.copy()
        
        # Add temporal features
        if "created_at" in tasks_enhanced.columns:
            tasks_enhanced = self.extract_temporal_features(tasks_enhanced, "created_at")
        
        if "due_date" in tasks_enhanced.columns:
            tasks_enhanced = self.extract_temporal_features(tasks_enhanced, "due_date")
        
        # Add text features if description column exists
        if "description" in tasks_df.columns:
            tasks_enhanced = self.extract_text_features(tasks_enhanced, "description", max_features=50)
        
        # Add user activity features
        tasks_enhanced = self.create_user_activity_features(tasks_enhanced, activities_df)
        
        # Add user preference features
        tasks_enhanced = self.create_user_preference_features(tasks_enhanced, preferences_df)
        
        # Enhanced recommendations features
        recommendations_enhanced = recommendations_df.copy()
        
        # Add temporal features
        if "timestamp" in recommendations_enhanced.columns:
            recommendations_enhanced = self.extract_temporal_features(recommendations_enhanced, "timestamp")
        
        # Add user preference features
        recommendations_enhanced = self.create_user_preference_features(recommendations_enhanced, preferences_df)
        
        # Create interaction features for numerical columns
        num_cols = recommendations_enhanced.select_dtypes(include=["int64", "float64"]).columns.tolist()
        recommendations_enhanced = self.create_interaction_features(recommendations_enhanced, num_cols)
        
        # Save transformers
        self.save_transformers()
        
        return {
            "tasks": tasks_enhanced,
            "recommendations": recommendations_enhanced
        }


if __name__ == "__main__":
    # Load processed data
    from src.data_processing.data_processor import DataProcessor
    
    processor = DataProcessor()
    tasks_df = processor.load_processed_data("processed_tasks.csv")
    activities_df = processor.load_processed_data("processed_activities.csv")
    recommendations_df = processor.load_processed_data("processed_recommendations.csv")
    
    # Load user preferences
    preferences_data = processor.load_json_data(os.path.join(processor.synthetic_data_dir, "user_preferences.json"))
    preferences_df = pd.DataFrame(preferences_data)
    
    # Create feature engineer
    engineer = FeatureEngineer()
    
    # Create enhanced features
    enhanced_dfs = engineer.create_all_features(tasks_df, activities_df, recommendations_df, preferences_df)
    
    # Save enhanced features
    for name, df in enhanced_dfs.items():
        processor.save_processed_data(df, f"enhanced_{name}.csv")
        print(f"Saved enhanced {name} features with shape {df.shape}")