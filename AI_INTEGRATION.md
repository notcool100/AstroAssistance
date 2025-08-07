# AstroAssistance AI Integration

This document provides an overview of the AI integration in AstroAssistance, explaining how the AI models work, how they're trained, and how they're integrated with the backend and frontend.

## Architecture Overview

The AI system in AstroAssistance consists of several components:

1. **AI Models**: Machine learning models for task completion prediction and task duration prediction
2. **AI Service**: Backend service that interfaces with the AI models
3. **Python Scripts**: Scripts for training models and generating predictions
4. **Database Integration**: Storage of AI-generated recommendations and user feedback
5. **Frontend Components**: UI for displaying and interacting with AI recommendations

## AI Models

### Task Completion Predictor

This model predicts whether a task will be completed on time based on:
- Task priority
- Task category
- Time to deadline
- Estimated duration

The model is implemented as a Random Forest Classifier with preprocessing for categorical and numerical features.

### Task Duration Predictor

This model predicts how long a task will actually take to complete based on:
- Task priority
- Task category
- Estimated duration
- Number of tags

The model is implemented as a Gradient Boosting Regressor with preprocessing for categorical and numerical features.

## Training Process

The AI models are trained using:

1. **Historical Task Data**: Completed tasks with actual durations and completion times
2. **User Feedback**: Ratings and comments on AI-generated recommendations
3. **Synthetic Data**: Generated data to supplement real data during initial deployment

Training can be triggered in two ways:

1. **Scheduled Training**: Automatically runs daily at 2 AM using the `schedule_ai_training.ts` script
2. **Manual Training**: Can be triggered via the `/api/ai/train` endpoint

The training process involves:
1. Exporting data from the database
2. Preprocessing the data
3. Training the models
4. Evaluating model performance
5. Saving the trained models

## Integration with Backend

The AI is integrated with the backend through:

1. **AI Service**: `ai.service.ts` provides methods for prediction and recommendation generation
2. **Model Integration**: Task and Recommendation models use the AI service
3. **API Endpoints**: New endpoints for AI-related operations
4. **Feedback Collection**: User feedback is stored and used for model improvement

### Key Files:

- `/backend/src/services/ai.service.ts`: Main service for AI functionality
- `/backend/ai/predict_task.py`: Python script for task prediction
- `/backend/ai/generate_recommendations.py`: Python script for recommendation generation
- `/backend/ai/train_models.py`: Python script for model training
- `/backend/scripts/export_data_for_training.ts`: Script to export data for training
- `/backend/scripts/schedule_ai_training.ts`: Script to schedule regular training

## Integration with Frontend

The frontend displays AI-generated recommendations and collects user feedback:

1. **Recommendations Page**: Shows personalized AI recommendations
2. **Feedback Mechanisms**: Users can rate recommendations and mark them as implemented
3. **Generate Button**: Users can request new AI-generated recommendations

## Continuous Learning

The system implements continuous learning through:

1. **Feedback Collection**: User interactions provide feedback for model improvement
2. **Regular Retraining**: Models are retrained daily with new data
3. **Performance Monitoring**: Model metrics are tracked over time

## Setup and Configuration

To set up the AI integration:

1. Ensure Python 3.8+ is installed
2. Install required Python packages:
   ```
   pip install scikit-learn pandas numpy joblib
   ```
3. Create necessary directories:
   ```
   mkdir -p models data/raw data/processed
   ```
4. Run initial training:
   ```
   cd backend
   npx ts-node scripts/export_data_for_training.ts
   python ai/train_models.py
   ```
5. Set up scheduled training:
   ```
   npx ts-node scripts/schedule_ai_training.ts
   ```

## API Endpoints

The following endpoints are available for AI functionality:

- `POST /api/ai/train`: Trigger AI model training
- `POST /api/ai/feedback`: Submit feedback for AI recommendations
- `GET /api/ai/status`: Get AI system status
- `GET /api/recommendations/generate`: Generate new AI recommendations

## Troubleshooting

Common issues and solutions:

1. **Models Not Found**: Ensure the models directory exists and initial training has been run
2. **Training Failures**: Check logs for errors and ensure data is available
3. **Prediction Errors**: The system includes fallbacks for when predictions fail
4. **Performance Issues**: If models perform poorly, try with more training data

## Future Improvements

Planned enhancements to the AI system:

1. **Advanced Models**: Implement deep learning models for more complex patterns
2. **More Features**: Add more features to improve prediction accuracy
3. **A/B Testing**: Test different recommendation strategies with users
4. **Explainable AI**: Provide better explanations for recommendations
5. **Real-time Predictions**: Move from batch to real-time predictions