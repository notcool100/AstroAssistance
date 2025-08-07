import express from 'express';
import { body } from 'express-validator';
import { Task } from '../models';
import { protect } from '../middleware/auth';
import { validateRequest } from '../middleware/validate';
import { logger } from '../utils/logger';

const router = express.Router();

/**
 * @route   POST /api/tasks
 * @desc    Create a new task
 * @access  Private
 */
router.post(
  '/',
  [
    protect,
    body('title').notEmpty().withMessage('Title is required'),
    body('category').notEmpty().withMessage('Category is required'),
    body('priority').notEmpty().withMessage('Priority is required'),
    body('dueDate').optional().custom(value => {
      // Accept both ISO8601 and YYYY-MM-DD formats
      if (!value) return true;
      
      // Try to create a valid date from the input
      const date = new Date(value);
      if (isNaN(date.getTime())) {
        throw new Error('Invalid date format');
      }
      return true;
    }),
    body('estimatedDuration').optional().isNumeric().withMessage('Duration must be a number'),
    body('tags').optional().isArray().withMessage('Tags must be an array'),
    validateRequest,
  ],
  async (req, res) => {
    try {
      const task = await Task.create(req.user.id, req.body);
      res.status(201).json(task);
    } catch (error: any) {
      logger.error(`Create task error: ${error.message}`);
      res.status(500).json({ message: 'Server error' });
    }
  }
);

/**
 * @route   GET /api/tasks
 * @desc    Get all tasks for a user
 * @access  Private
 */
router.get('/', protect, async (req, res) => {
  try {
    const result = await Task.getAll(req.user.id, req.query);
    res.json(result);
  } catch (error: any) {
    logger.error(`Get tasks error: ${error.message}`);
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * @route   GET /api/tasks/:id
 * @desc    Get a task by ID
 * @access  Private
 */
router.get('/:id', protect, async (req, res) => {
  try {
    const task = await Task.getById(req.params.id, req.user.id);
    res.json(task);
  } catch (error: any) {
    logger.error(`Get task error: ${error.message}`);
    if (error.message === 'Task not found') {
      return res.status(404).json({ message: error.message });
    }
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * @route   PUT /api/tasks/:id
 * @desc    Update a task
 * @access  Private
 */
router.put(
  '/:id',
  [
    protect,
    body('title').optional(),
    body('description').optional(),
    body('category').optional(),
    body('priority').optional(),
    body('dueDate').optional().custom(value => {
      // Accept both ISO8601 and YYYY-MM-DD formats
      if (!value) return true;
      
      // Try to create a valid date from the input
      const date = new Date(value);
      if (isNaN(date.getTime())) {
        throw new Error('Invalid date format');
      }
      return true;
    }),
    body('estimatedDuration').optional().isNumeric().withMessage('Duration must be a number'),
    body('tags').optional().isArray().withMessage('Tags must be an array'),
    validateRequest,
  ],
  async (req, res) => {
    try {
      const task = await Task.update(req.params.id, req.user.id, req.body);
      res.json(task);
    } catch (error: any) {
      logger.error(`Update task error: ${error.message}`);
      if (error.message === 'Task not found') {
        return res.status(404).json({ message: error.message });
      }
      res.status(500).json({ message: 'Server error' });
    }
  }
);

/**
 * @route   PUT /api/tasks/:id/complete
 * @desc    Mark a task as completed
 * @access  Private
 */
router.put('/:id/complete', protect, async (req, res) => {
  try {
    const completed = req.body.completed !== false; // Default to true if not specified
    const task = await Task.complete(req.params.id, req.user.id, completed);
    res.json(task);
  } catch (error: any) {
    logger.error(`Complete task error: ${error.message}`);
    if (error.message === 'Task not found') {
      return res.status(404).json({ message: error.message });
    }
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * @route   DELETE /api/tasks/:id
 * @desc    Delete a task
 * @access  Private
 */
router.delete('/:id', protect, async (req, res) => {
  try {
    const result = await Task.delete(req.params.id, req.user.id);
    res.json(result);
  } catch (error: any) {
    logger.error(`Delete task error: ${error.message}`);
    if (error.message === 'Task not found') {
      return res.status(404).json({ message: error.message });
    }
    res.status(500).json({ message: 'Server error' });
  }
});

export default router;