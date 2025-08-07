import express from 'express';
import { body } from 'express-validator';
import { LearningFeedback } from '../models';
import { protect } from '../middleware/auth';
import { validateRequest } from '../middleware/validate';
import { logger } from '../utils/logger';

const router = express.Router();

/**
 * @route   POST /api/feedback
 * @desc    Create new feedback
 * @access  Private
 */
router.post(
  '/',
  [
    protect,
    body('rating').isInt({ min: 1, max: 5 }).withMessage('Rating must be between 1 and 5'),
    body('recommendationId').optional().isUUID().withMessage('Invalid recommendation ID'),
    validateRequest,
  ],
  async (req, res) => {
    try {
      const feedback = await LearningFeedback.create(req.user.id, req.body);
      res.status(201).json(feedback);
    } catch (error: any) {
      logger.error(`Create feedback error: ${error.message}`);
      if (error.message === 'Recommendation not found') {
        return res.status(404).json({ message: error.message });
      }
      res.status(500).json({ message: 'Server error' });
    }
  }
);

/**
 * @route   GET /api/feedback
 * @desc    Get all feedback for a user
 * @access  Private
 */
router.get('/', protect, async (req, res) => {
  try {
    const result = await LearningFeedback.getAll(req.user.id, req.query);
    res.json(result);
  } catch (error: any) {
    logger.error(`Get feedbacks error: ${error.message}`);
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * @route   GET /api/feedback/:id
 * @desc    Get feedback by ID
 * @access  Private
 */
router.get('/:id', protect, async (req, res) => {
  try {
    const feedback = await LearningFeedback.getById(req.params.id, req.user.id);
    res.json(feedback);
  } catch (error: any) {
    logger.error(`Get feedback error: ${error.message}`);
    if (error.message === 'Feedback not found') {
      return res.status(404).json({ message: error.message });
    }
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * @route   DELETE /api/feedback/:id
 * @desc    Delete feedback
 * @access  Private
 */
router.delete('/:id', protect, async (req, res) => {
  try {
    const result = await LearningFeedback.delete(req.params.id, req.user.id);
    res.json(result);
  } catch (error: any) {
    logger.error(`Delete feedback error: ${error.message}`);
    if (error.message === 'Feedback not found') {
      return res.status(404).json({ message: error.message });
    }
    res.status(500).json({ message: 'Server error' });
  }
});

export default router;