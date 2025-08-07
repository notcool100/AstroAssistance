"""
Data processing utilities for AstroAssistance.
"""
import os
import json
import pickle
from typing import Dict, List, Any, Tuple, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

from src.core.config import config_manager
from src.core.logger import app_logger
from src.core.data_types import (
    Task, TaskPriority, TaskStatus, TaskCategory,
    Reminder, Goal, UserPreference, UserActivity,
    LearningFeedback, Recommendation, RecurrencePattern
)


class DataProcessor:
    """Processes data for model training and inference."""
    
    def __init__(self):
        """Initialize the data processor."""
        # Get project root directory
        self.project_root = Path(__file__).parent.parent.parent
        self.raw_data_dir = os.path.join(self.project_root, "data", "raw")
        self.processed_data_dir = os.path.join(self.project_root, "data", "processed")
        self.synthetic_data_dir = os.path.join(self.project_root, "data", "synthetic")
        
        # Create directories if they don't exist
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        # Initialize preprocessing components
        self.scalers = {}
        self.encoders = {}
    
    def load_json_data(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load data from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            List of dictionaries containing the data
        """
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            return data
        except Exception as e:
            app_logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise
    
    def save_processed_data(self, data: Any, filename: str) -> str:
        """
        Save processed data to a file.
        
        Args:
            data: Data to save
            filename: Name of the file
            
        Returns:
            Path to the saved file
        """
        file_path = os.path.join(self.processed_data_dir, filename)
        
        # Determine file format based on extension
        if filename.endswith(".csv"):
            data.to_csv(file_path, index=False)
        elif filename.endswith(".pkl"):
            with open(file_path, "wb") as f:
                pickle.dump(data, f)
        elif filename.endswith(".json"):
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        elif filename.endswith(".pt") or filename.endswith(".pth"):
            torch.save(data, file_path)
        else:
            app_logger.warning(f"Unknown file format for {filename}, saving as pickle")
            with open(file_path, "wb") as f:
                pickle.dump(data, f)
        
        app_logger.info(f"Saved processed data to {file_path}")
        return file_path
    
    def load_processed_data(self, filename: str) -> Any:
        """
        Load processed data from a file.
        
        Args:
            filename: Name of the file
            
        Returns:
            Loaded data
        """
        file_path = os.path.join(self.processed_data_dir, filename)
        
        if not os.path.exists(file_path):
            app_logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine file format based on extension
        if file_path.endswith(".csv"):
            return pd.read_csv(file_path)
        elif file_path.endswith(".pkl"):
            with open(file_path, "rb") as f:
                return pickle.load(f)
        elif file_path.endswith(".json"):
            with open(file_path, "r") as f:
                return json.load(f)
        elif file_path.endswith(".pt") or file_path.endswith(".pth"):
            return torch.load(file_path)
        else:
            app_logger.warning(f"Unknown file format for {file_path}, trying pickle")
            with open(file_path, "rb") as f:
                return pickle.load(f)
    
    def process_tasks(self, tasks_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Process task data.
        
        Args:
            tasks_data: List of task dictionaries
            
        Returns:
            DataFrame of processed task data
        """
        app_logger.info(f"Processing {len(tasks_data)} tasks")
        
        # Convert to DataFrame
        df = pd.DataFrame(tasks_data)
        
        # Convert string dates to datetime
        date_columns = ["due_date", "created_at", "updated_at", "completed_at", "recurrence_end_date"]
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        
        # Extract features from dates
        if "created_at" in df.columns:
            df["created_day_of_week"] = df["created_at"].dt.dayofweek
            df["created_hour"] = df["created_at"].dt.hour
        
        if "due_date" in df.columns and "created_at" in df.columns:
            # Calculate days until due
            df["days_until_due"] = (df["due_date"] - df["created_at"]).dt.total_seconds() / (24 * 3600)
            df["days_until_due"] = df["days_until_due"].fillna(-1)  # No due date
        
        if "completed_at" in df.columns and "created_at" in df.columns:
            # Calculate completion time
            df["completion_time_days"] = (df["completed_at"] - df["created_at"]).dt.total_seconds() / (24 * 3600)
            df["completion_time_days"] = df["completion_time_days"].fillna(-1)  # Not completed
            
            # Calculate if completed on time
            df["completed_on_time"] = np.where(
                (df["completed_at"].notna()) & (df["due_date"].notna()),
                df["completed_at"] <= df["due_date"],
                np.nan
            )
        
        # Convert categorical columns
        cat_columns = ["priority", "status", "category", "recurrence"]
        for col in cat_columns:
            if col in df.columns:
                df[col] = df[col].astype("category")
        
        # Handle tags
        if "tags" in df.columns:
            # Count tags
            df["tag_count"] = df["tags"].apply(lambda x: len(x) if isinstance(x, list) else 0)
            
            # Get most common tags
            all_tags = []
            for tags in df["tags"]:
                if isinstance(tags, list):
                    all_tags.extend(tags)
            
            if all_tags:
                from collections import Counter
                top_tags = [tag for tag, _ in Counter(all_tags).most_common(10)]
                
                # Create binary features for top tags
                for tag in top_tags:
                    df[f"has_tag_{tag}"] = df["tags"].apply(lambda x: tag in x if isinstance(x, list) else False)
        
        # Handle subtasks
        if "subtasks" in df.columns:
            df["subtask_count"] = df["subtasks"].apply(lambda x: len(x) if isinstance(x, list) else 0)
        
        # Drop columns that are not useful for modeling
        cols_to_drop = ["id", "description", "notes", "attachments", "tags", "subtasks"]
        df = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)
        
        return df
    
    def process_user_activities(self, activities_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Process user activity data.
        
        Args:
            activities_data: List of user activity dictionaries
            
        Returns:
            DataFrame of processed user activity data
        """
        app_logger.info(f"Processing {len(activities_data)} user activities")
        
        # Convert to DataFrame
        df = pd.DataFrame(activities_data)
        
        # Convert string dates to datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df["day_of_week"] = df["timestamp"].dt.dayofweek
            df["hour_of_day"] = df["timestamp"].dt.hour
        
        # Extract activity type
        if "activity_type" in df.columns:
            df["activity_type"] = df["activity_type"].astype("category")
            
            # Create binary features for activity types
            activity_types = df["activity_type"].unique()
            for activity_type in activity_types:
                df[f"is_{activity_type}"] = (df["activity_type"] == activity_type)
        
        # Extract context features
        if "context" in df.columns:
            # Extract device
            df["device"] = df["context"].apply(
                lambda x: x.get("device") if isinstance(x, dict) else None
            )
            
            # Extract location
            df["location"] = df["context"].apply(
                lambda x: x.get("location") if isinstance(x, dict) else None
            )
            
            # Convert to categorical
            if "device" in df.columns:
                df["device"] = df["device"].astype("category")
            
            if "location" in df.columns:
                df["location"] = df["location"].astype("category")
        
        # Drop complex columns
        cols_to_drop = ["id", "activity_data", "context"]
        df = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)
        
        return df
    
    def process_recommendations(self, recommendations_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Process recommendation data.
        
        Args:
            recommendations_data: List of recommendation dictionaries
            
        Returns:
            DataFrame of processed recommendation data
        """
        app_logger.info(f"Processing {len(recommendations_data)} recommendations")
        
        # Convert to DataFrame
        df = pd.DataFrame(recommendations_data)
        
        # Convert string dates to datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df["day_of_week"] = df["timestamp"].dt.dayofweek
            df["hour_of_day"] = df["timestamp"].dt.hour
        
        # Extract recommendation type
        if "recommendation_type" in df.columns:
            df["recommendation_type"] = df["recommendation_type"].astype("category")
            
            # Create binary features for recommendation types
            rec_types = df["recommendation_type"].unique()
            for rec_type in rec_types:
                df[f"is_{rec_type}"] = (df["recommendation_type"] == rec_type)
        
        # Drop complex columns
        cols_to_drop = ["id", "content", "explanation"]
        df = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)
        
        return df
    
    def process_learning_feedback(self, feedback_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Process learning feedback data.
        
        Args:
            feedback_data: List of feedback dictionaries
            
        Returns:
            DataFrame of processed feedback data
        """
        app_logger.info(f"Processing {len(feedback_data)} feedback entries")
        
        # Convert to DataFrame
        df = pd.DataFrame(feedback_data)
        
        # Convert string dates to datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df["day_of_week"] = df["timestamp"].dt.dayofweek
            df["hour_of_day"] = df["timestamp"].dt.hour
        
        # Extract feedback type
        if "feedback_type" in df.columns:
            df["feedback_type"] = df["feedback_type"].astype("category")
            
            # Create binary features for feedback types
            feedback_types = df["feedback_type"].unique()
            for feedback_type in feedback_types:
                df[f"is_{feedback_type}"] = (df["feedback_type"] == feedback_type)
        
        # Extract context features
        if "context" in df.columns:
            # Extract device
            df["device"] = df["context"].apply(
                lambda x: x.get("device") if isinstance(x, dict) else None
            )
            
            # Extract location
            df["location"] = df["context"].apply(
                lambda x: x.get("location") if isinstance(x, dict) else None
            )
            
            # Convert to categorical
            if "device" in df.columns:
                df["device"] = df["device"].astype("category")
            
            if "location" in df.columns:
                df["location"] = df["location"].astype("category")
        
        # Has user comments
        if "user_comments" in df.columns:
            df["has_comments"] = df["user_comments"].notna()
        
        # Drop complex columns
        cols_to_drop = ["id", "original_recommendation", "modified_recommendation", "context", "user_comments"]
        df = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)
        
        return df
    
    def prepare_task_completion_data(self, tasks_df: pd.DataFrame, activities_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare data for task completion prediction.
        
        Args:
            tasks_df: DataFrame of processed task data
            activities_df: DataFrame of processed user activity data
            
        Returns:
            Tuple of (features_df, target_array)
        """
        app_logger.info("Preparing task completion prediction data")
        
        # Filter tasks with completion status
        tasks_with_status = tasks_df.dropna(subset=["status"])
        
        # Create target variable: 1 if completed, 0 otherwise
        y = (tasks_with_status["status"] == "completed").astype(int).values
        
        # Select features for prediction
        feature_cols = [
            "priority", "category", "estimated_duration", "tag_count",
            "created_day_of_week", "created_hour", "days_until_due"
        ]
        
        # Add subtask count if available
        if "subtask_count" in tasks_with_status.columns:
            feature_cols.append("subtask_count")
        
        # Filter to only include columns that exist
        feature_cols = [col for col in feature_cols if col in tasks_with_status.columns]
        
        X = tasks_with_status[feature_cols].copy()
        
        # Handle categorical features
        cat_features = ["priority", "category"]
        cat_features = [col for col in cat_features if col in X.columns]
        
        # One-hot encode categorical features
        if cat_features:
            encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
            encoded_cats = encoder.fit_transform(X[cat_features])
            
            # Create DataFrame with encoded features
            encoded_df = pd.DataFrame(
                encoded_cats,
                columns=encoder.get_feature_names_out(cat_features),
                index=X.index
            )
            
            # Drop original categorical columns and join encoded ones
            X = X.drop(cat_features, axis=1).join(encoded_df)
            
            # Save encoder
            self.encoders["task_completion"] = encoder
        
        # Handle missing values
        X = X.fillna(-1)
        
        # Scale numerical features
        num_features = X.select_dtypes(include=["int64", "float64"]).columns
        if not num_features.empty:
            scaler = StandardScaler()
            X[num_features] = scaler.fit_transform(X[num_features])
            
            # Save scaler
            self.scalers["task_completion"] = scaler
        
        return X, y
    
    def prepare_recommendation_acceptance_data(self, recommendations_df: pd.DataFrame, feedback_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare data for recommendation acceptance prediction.
        
        Args:
            recommendations_df: DataFrame of processed recommendation data
            feedback_df: DataFrame of processed feedback data
            
        Returns:
            Tuple of (features_df, target_array)
        """
        app_logger.info("Preparing recommendation acceptance prediction data")
        
        # Merge recommendations with feedback
        if "recommendation_id" in feedback_df.columns and "id" in recommendations_df.columns:
            # Rename id column in recommendations for merging
            recommendations_df = recommendations_df.rename(columns={"id": "recommendation_id"})
            
            # Merge dataframes
            merged_df = pd.merge(
                recommendations_df,
                feedback_df,
                on="recommendation_id",
                how="left",
                suffixes=("_rec", "_feed")
            )
        else:
            # If we can't merge, just use recommendations
            merged_df = recommendations_df.copy()
            # Assume is_applied is our target
            if "is_applied" not in merged_df.columns:
                app_logger.warning("No 'is_applied' column found in recommendations data")
                merged_df["is_applied"] = np.random.choice([0, 1], size=len(merged_df))
        
        # Create target variable: 1 if accepted or modified, 0 if rejected or no feedback
        if "feedback_type" in merged_df.columns:
            y = ((merged_df["feedback_type"] == "accepted") | (merged_df["feedback_type"] == "modified")).astype(int).values
        else:
            # Use is_applied as target if feedback_type not available
            y = merged_df["is_applied"].astype(int).values
        
        # Select features for prediction
        feature_cols = [
            "recommendation_type", "confidence_score", "day_of_week", "hour_of_day"
        ]
        
        # Add binary recommendation type columns if available
        type_cols = [col for col in merged_df.columns if col.startswith("is_") and col != "is_applied"]
        feature_cols.extend(type_cols)
        
        # Filter to only include columns that exist
        feature_cols = [col for col in feature_cols if col in merged_df.columns]
        
        X = merged_df[feature_cols].copy()
        
        # Handle categorical features
        cat_features = ["recommendation_type"]
        cat_features = [col for col in cat_features if col in X.columns]
        
        # One-hot encode categorical features
        if cat_features:
            encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
            encoded_cats = encoder.fit_transform(X[cat_features])
            
            # Create DataFrame with encoded features
            encoded_df = pd.DataFrame(
                encoded_cats,
                columns=encoder.get_feature_names_out(cat_features),
                index=X.index
            )
            
            # Drop original categorical columns and join encoded ones
            X = X.drop(cat_features, axis=1).join(encoded_df)
            
            # Save encoder
            self.encoders["recommendation_acceptance"] = encoder
        
        # Handle missing values
        X = X.fillna(-1)
        
        # Scale numerical features
        num_features = X.select_dtypes(include=["int64", "float64"]).columns
        if not num_features.empty:
            scaler = StandardScaler()
            X[num_features] = scaler.fit_transform(X[num_features])
            
            # Save scaler
            self.scalers["recommendation_acceptance"] = scaler
        
        return X, y
    
    def prepare_task_duration_data(self, tasks_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare data for task duration prediction.
        
        Args:
            tasks_df: DataFrame of processed task data
            
        Returns:
            Tuple of (features_df, target_array)
        """
        app_logger.info("Preparing task duration prediction data")
        
        # Filter tasks with actual duration
        tasks_with_duration = tasks_df.dropna(subset=["actual_duration"])
        
        # Create target variable: actual duration
        y = tasks_with_duration["actual_duration"].values
        
        # Select features for prediction
        feature_cols = [
            "priority", "category", "estimated_duration", "tag_count",
            "created_day_of_week", "created_hour"
        ]
        
        # Add subtask count if available
        if "subtask_count" in tasks_with_duration.columns:
            feature_cols.append("subtask_count")
        
        # Filter to only include columns that exist
        feature_cols = [col for col in feature_cols if col in tasks_with_duration.columns]
        
        X = tasks_with_duration[feature_cols].copy()
        
        # Handle categorical features
        cat_features = ["priority", "category"]
        cat_features = [col for col in cat_features if col in X.columns]
        
        # One-hot encode categorical features
        if cat_features:
            encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
            encoded_cats = encoder.fit_transform(X[cat_features])
            
            # Create DataFrame with encoded features
            encoded_df = pd.DataFrame(
                encoded_cats,
                columns=encoder.get_feature_names_out(cat_features),
                index=X.index
            )
            
            # Drop original categorical columns and join encoded ones
            X = X.drop(cat_features, axis=1).join(encoded_df)
            
            # Save encoder
            self.encoders["task_duration"] = encoder
        
        # Handle missing values
        X = X.fillna(-1)
        
        # Scale numerical features
        num_features = X.select_dtypes(include=["int64", "float64"]).columns
        if not num_features.empty:
            scaler = StandardScaler()
            X[num_features] = scaler.fit_transform(X[num_features])
            
            # Save scaler
            self.scalers["task_duration"] = scaler
        
        # Scale target variable (log transform to handle skewness)
        y = np.log1p(y)
        
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: np.ndarray, test_size: float = 0.2, val_size: float = 0.15) -> Tuple:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Feature DataFrame
            y: Target array
            test_size: Proportion of data to use for testing
            val_size: Proportion of data to use for validation
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: separate test set
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Second split: separate validation set from training set
        # Adjust validation size to account for the test split
        adjusted_val_size = val_size / (1 - test_size)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=adjusted_val_size, random_state=42
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_preprocessors(self) -> None:
        """Save all preprocessors (scalers and encoders)."""
        # Save scalers
        for name, scaler in self.scalers.items():
            self.save_processed_data(scaler, f"{name}_scaler.pkl")
        
        # Save encoders
        for name, encoder in self.encoders.items():
            self.save_processed_data(encoder, f"{name}_encoder.pkl")
    
    def load_preprocessors(self) -> None:
        """Load all preprocessors (scalers and encoders)."""
        # Get all preprocessor files
        preprocessor_files = [f for f in os.listdir(self.processed_data_dir) if f.endswith("_scaler.pkl") or f.endswith("_encoder.pkl")]
        
        for file in preprocessor_files:
            try:
                preprocessor = self.load_processed_data(file)
                
                if file.endswith("_scaler.pkl"):
                    name = file.replace("_scaler.pkl", "")
                    self.scalers[name] = preprocessor
                elif file.endswith("_encoder.pkl"):
                    name = file.replace("_encoder.pkl", "")
                    self.encoders[name] = preprocessor
            except Exception as e:
                app_logger.error(f"Error loading preprocessor {file}: {str(e)}")
    
    def prepare_all_datasets(self) -> Dict[str, str]:
        """
        Prepare all datasets for model training.
        
        Returns:
            Dictionary mapping dataset name to file path
        """
        app_logger.info("Preparing all datasets")
        
        # Load synthetic data
        tasks_data = self.load_json_data(os.path.join(self.synthetic_data_dir, "tasks.json"))
        activities_data = self.load_json_data(os.path.join(self.synthetic_data_dir, "user_activities.json"))
        recommendations_data = self.load_json_data(os.path.join(self.synthetic_data_dir, "recommendations.json"))
        feedback_data = self.load_json_data(os.path.join(self.synthetic_data_dir, "learning_feedback.json"))
        
        # Process data
        tasks_df = self.process_tasks(tasks_data)
        activities_df = self.process_user_activities(activities_data)
        recommendations_df = self.process_recommendations(recommendations_data)
        feedback_df = self.process_learning_feedback(feedback_data)
        
        # Save processed DataFrames
        self.save_processed_data(tasks_df, "processed_tasks.csv")
        self.save_processed_data(activities_df, "processed_activities.csv")
        self.save_processed_data(recommendations_df, "processed_recommendations.csv")
        self.save_processed_data(feedback_df, "processed_feedback.csv")
        
        # Prepare task completion data
        X_completion, y_completion = self.prepare_task_completion_data(tasks_df, activities_df)
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X_completion, y_completion)
        
        # Save task completion datasets
        self.save_processed_data((X_train, y_train), "task_completion_train.pkl")
        self.save_processed_data((X_val, y_val), "task_completion_val.pkl")
        self.save_processed_data((X_test, y_test), "task_completion_test.pkl")
        
        # Prepare recommendation acceptance data
        X_acceptance, y_acceptance = self.prepare_recommendation_acceptance_data(recommendations_df, feedback_df)
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X_acceptance, y_acceptance)
        
        # Save recommendation acceptance datasets
        self.save_processed_data((X_train, y_train), "recommendation_acceptance_train.pkl")
        self.save_processed_data((X_val, y_val), "recommendation_acceptance_val.pkl")
        self.save_processed_data((X_test, y_test), "recommendation_acceptance_test.pkl")
        
        # Prepare task duration data
        X_duration, y_duration = self.prepare_task_duration_data(tasks_df)
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X_duration, y_duration)
        
        # Save task duration datasets
        self.save_processed_data((X_train, y_train), "task_duration_train.pkl")
        self.save_processed_data((X_val, y_val), "task_duration_val.pkl")
        self.save_processed_data((X_test, y_test), "task_duration_test.pkl")
        
        # Save preprocessors
        self.save_preprocessors()
        
        return {
            "task_completion": os.path.join(self.processed_data_dir, "task_completion_train.pkl"),
            "recommendation_acceptance": os.path.join(self.processed_data_dir, "recommendation_acceptance_train.pkl"),
            "task_duration": os.path.join(self.processed_data_dir, "task_duration_train.pkl")
        }


class TaskDataset(Dataset):
    """PyTorch dataset for task data."""
    
    def __init__(self, X: Union[pd.DataFrame, np.ndarray], y: np.ndarray):
        """
        Initialize the dataset.
        
        Args:
            X: Feature matrix
            y: Target array
        """
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            self.X = X.values.astype(np.float32)
        else:
            self.X = X.astype(np.float32)
        
        self.y = y.astype(np.float32)
    
    def __len__(self) -> int:
        """Get the number of samples."""
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (features, target)
        """
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)


def create_data_loaders(X_train: pd.DataFrame, y_train: np.ndarray, 
                        X_val: pd.DataFrame, y_val: np.ndarray,
                        batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training and validation.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        batch_size: Batch size
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = TaskDataset(X_train, y_train)
    val_dataset = TaskDataset(X_val, y_val)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader


if __name__ == "__main__":
    processor = DataProcessor()
    dataset_paths = processor.prepare_all_datasets()
    print(f"Prepared datasets: {dataset_paths}")