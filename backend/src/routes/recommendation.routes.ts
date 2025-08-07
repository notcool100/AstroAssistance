import express from 'express';
import { body, query } from 'express-validator';
import { Recommendation } from '../models';
import { protect } from '../middleware/auth';
import { validateRequest } from '../middleware/validate';
import { logger } from '../utils/logger';

const router = express.Router();

/**
 * @route   GET /api/recommendations
 * @desc    Get all recommendations for a user
 * @access  Private
 */
router.get('/', protect, async (req, res) => {
  try {
    const result = await Recommendation.getAll(req.user.id, req.query);
    res.json(result);
  } catch (error: any) {
    logger.error(`Get recommendations error: ${error.message}`);
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * @route   GET /api/recommendations/generate
 * @desc    Generate recommendations for a user
 * @access  Private
 */
router.get(
  '/generate',
  [
    protect,
    query('count').optional().isInt({ min: 1, max: 10 }).withMessage('Count must be between 1 and 10'),
    validateRequest,
  ],
  async (req, res) => {
    try {
      const count = parseInt(req.query.count as string) || 3;
      const recommendations = await Recommendation.generate(req.user.id, count);
      res.json(recommendations);
    } catch (error: any) {
      logger.error(`Generate recommendations error: ${error.message}`);
      res.status(500).json({ message: 'Server error' });
    }
  }
);

/**
 * @route   GET /api/recommendations/:id
 * @desc    Get a recommendation by ID
 * @access  Private
 */
router.get('/:id', protect, async (req, res) => {
  try {
    const recommendation = await Recommendation.getById(req.params.id, req.user.id);
    res.json(recommendation);
  } catch (error: any) {
    logger.error(`Get recommendation error: ${error.message}`);
    if (error.message === 'Recommendation not found') {
      return res.status(404).json({ message: error.message });
    }
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * @route   POST /api/recommendations
 * @desc    Create a new recommendation (manual)
 * @access  Private
 */
router.post(
  '/',
  [
    protect,
    body('type').notEmpty().withMessage('Type is required'),
    body('content').notEmpty().withMessage('Content is required'),
    validateRequest,
  ],
  async (req, res) => {
    try {
      const recommendation = await Recommendation.create(req.user.id, req.body);
      res.status(201).json(recommendation);
    } catch (error: any) {
      logger.error(`Create recommendation error: ${error.message}`);
      res.status(500).json({ message: 'Server error' });
    }
  }
);

/**
 * @route   PUT /api/recommendations/:id/apply
 * @desc    Mark a recommendation as applied
 * @access  Private
 */
router.put('/:id/apply', protect, async (req, res) => {
  try {
    const applied = req.body.applied !== false; // Default to true if not specified
    const recommendation = await Recommendation.apply(req.params.id, req.user.id, applied);
    res.json(recommendation);
  } catch (error: any) {
    logger.error(`Apply recommendation error: ${error.message}`);
    if (error.message === 'Recommendation not found') {
      return res.status(404).json({ message: error.message });
    }
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * @route   DELETE /api/recommendations/:id
 * @desc    Delete a recommendation
 * @access  Private
 */
router.delete('/:id', protect, async (req, res) => {
  try {
    const result = await Recommendation.delete(req.params.id, req.user.id);
    res.json(result);
  } catch (error: any) {
    logger.error(`Delete recommendation error: ${error.message}`);
    if (error.message === 'Recommendation not found') {
      return res.status(404).json({ message: error.message });
    }
    res.status(500).json({ message: 'Server error' });
  }
});

export default router;