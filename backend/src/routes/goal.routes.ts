import express from 'express';
import { body } from 'express-validator';
import { Goal } from '../models';
import { protect } from '../middleware/auth';
import { validateRequest } from '../middleware/validate';
import { logger } from '../utils/logger';

const router = express.Router();

/**
 * @route   POST /api/goals
 * @desc    Create a new goal
 * @access  Private
 */
router.post(
  '/',
  [
    protect,
    body('title').notEmpty().withMessage('Title is required'),
    body('targetDate').optional().isISO8601().withMessage('Invalid date format'),
    body('progress').optional().isFloat({ min: 0, max: 1 }).withMessage('Progress must be between 0 and 1'),
    validateRequest,
  ],
  async (req, res) => {
    try {
      const goal = await Goal.create(req.user.id, req.body);
      res.status(201).json(goal);
    } catch (error: any) {
      logger.error(`Create goal error: ${error.message}`);
      res.status(500).json({ message: 'Server error' });
    }
  }
);

/**
 * @route   GET /api/goals
 * @desc    Get all goals for a user
 * @access  Private
 */
router.get('/', protect, async (req, res) => {
  try {
    const result = await Goal.getAll(req.user.id, req.query);
    res.json(result);
  } catch (error: any) {
    logger.error(`Get goals error: ${error.message}`);
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * @route   GET /api/goals/:id
 * @desc    Get a goal by ID
 * @access  Private
 */
router.get('/:id', protect, async (req, res) => {
  try {
    const goal = await Goal.getById(req.params.id, req.user.id);
    res.json(goal);
  } catch (error: any) {
    logger.error(`Get goal error: ${error.message}`);
    if (error.message === 'Goal not found') {
      return res.status(404).json({ message: error.message });
    }
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * @route   PUT /api/goals/:id
 * @desc    Update a goal
 * @access  Private
 */
router.put(
  '/:id',
  [
    protect,
    body('title').optional(),
    body('description').optional(),
    body('targetDate').optional().isISO8601().withMessage('Invalid date format'),
    body('progress').optional().isFloat({ min: 0, max: 1 }).withMessage('Progress must be between 0 and 1'),
    validateRequest,
  ],
  async (req, res) => {
    try {
      const goal = await Goal.update(req.params.id, req.user.id, req.body);
      res.json(goal);
    } catch (error: any) {
      logger.error(`Update goal error: ${error.message}`);
      if (error.message === 'Goal not found') {
        return res.status(404).json({ message: error.message });
      }
      res.status(500).json({ message: 'Server error' });
    }
  }
);

/**
 * @route   PUT /api/goals/:id/complete
 * @desc    Mark a goal as completed
 * @access  Private
 */
router.put('/:id/complete', protect, async (req, res) => {
  try {
    const completed = req.body.completed !== false; // Default to true if not specified
    const goal = await Goal.complete(req.params.id, req.user.id, completed);
    res.json(goal);
  } catch (error: any) {
    logger.error(`Complete goal error: ${error.message}`);
    if (error.message === 'Goal not found') {
      return res.status(404).json({ message: error.message });
    }
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * @route   DELETE /api/goals/:id
 * @desc    Delete a goal
 * @access  Private
 */
router.delete('/:id', protect, async (req, res) => {
  try {
    const result = await Goal.delete(req.params.id, req.user.id);
    res.json(result);
  } catch (error: any) {
    logger.error(`Delete goal error: ${error.message}`);
    if (error.message === 'Goal not found') {
      return res.status(404).json({ message: error.message });
    }
    res.status(500).json({ message: 'Server error' });
  }
});

export default router;