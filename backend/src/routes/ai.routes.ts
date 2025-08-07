import express from 'express';
import { protect } from '../middleware/auth';
import { logger } from '../utils/logger';

// Import the AI service
const { AIService } = require('../services/ai.service');

const router = express.Router();

/**
 * @route   POST /api/ai/train
 * @desc    Train the AI models
 * @access  Private (Admin only)
 */
router.post('/train', protect, async (req, res) => {
  try {
    // In a real app, you would check if the user is an admin
    // For now, we'll allow any authenticated user to trigger training
    
    logger.info('Training AI models...');
    
    // Start the training process
    const result = await AIService.trainModels();
    
    res.json({ 
      success: true, 
      message: 'AI models training completed successfully',
      result 
    });
  } catch (error: any) {
    logger.error(`AI training error: ${error.message}`);
    res.status(500).json({ 
      success: false,
      message: 'AI training failed',
      error: error.message 
    });
  }
});

/**
 * @route   POST /api/ai/feedback
 * @desc    Submit feedback for AI recommendations
 * @access  Private
 */
router.post('/feedback', protect, async (req, res) => {
  try {
    const { recommendationId, rating, comment } = req.body;
    
    if (!recommendationId || !rating) {
      return res.status(400).json({ message: 'Recommendation ID and rating are required' });
    }
    
    // Process the feedback
    await AIService.processFeedback({
      userId: req.user.id,
      recommendationId,
      rating,
      comment
    });
    
    res.json({ 
      success: true, 
      message: 'Feedback submitted successfully' 
    });
  } catch (error: any) {
    logger.error(`AI feedback error: ${error.message}`);
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * @route   GET /api/ai/status
 * @desc    Get AI system status
 * @access  Private
 */
router.get('/status', protect, async (req, res) => {
  try {
    // Check if models exist
    const fs = require('fs');
    const path = require('path');
    
    const modelsDir = path.join(__dirname, '..', '..', '..', 'models');
    const completionModelPath = path.join(modelsDir, 'task_completion_predictor.joblib');
    const durationModelPath = path.join(modelsDir, 'task_duration_predictor.joblib');
    
    const completionModelExists = fs.existsSync(completionModelPath);
    const durationModelExists = fs.existsSync(durationModelPath);
    
    res.json({
      status: 'operational',
      models: {
        taskCompletionPredictor: {
          exists: completionModelExists,
          lastModified: completionModelExists ? 
            new Date(fs.statSync(completionModelPath).mtime).toISOString() : null
        },
        taskDurationPredictor: {
          exists: durationModelExists,
          lastModified: durationModelExists ? 
            new Date(fs.statSync(durationModelPath).mtime).toISOString() : null
        }
      }
    });
  } catch (error: any) {
    logger.error(`AI status error: ${error.message}`);
    res.status(500).json({ message: 'Server error' });
  }
});

export default router;