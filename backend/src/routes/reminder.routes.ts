import express from 'express';
import { body } from 'express-validator';
import { Reminder } from '../models';
import { protect } from '../middleware/auth';
import { validateRequest } from '../middleware/validate';
import { logger } from '../utils/logger';

const router = express.Router();

/**
 * @route   POST /api/reminders
 * @desc    Create a new reminder
 * @access  Private
 */
router.post(
  '/',
  [
    protect,
    body('title').notEmpty().withMessage('Title is required'),
    body('dueDate').notEmpty().isISO8601().withMessage('Valid due date is required'),
    validateRequest,
  ],
  async (req, res) => {
    try {
      const reminder = await Reminder.create(req.user.id, req.body);
      res.status(201).json(reminder);
    } catch (error: any) {
      logger.error(`Create reminder error: ${error.message}`);
      res.status(500).json({ message: 'Server error' });
    }
  }
);

/**
 * @route   GET /api/reminders
 * @desc    Get all reminders for a user
 * @access  Private
 */
router.get('/', protect, async (req, res) => {
  try {
    const result = await Reminder.getAll(req.user.id, req.query);
    res.json(result);
  } catch (error: any) {
    logger.error(`Get reminders error: ${error.message}`);
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * @route   GET /api/reminders/:id
 * @desc    Get a reminder by ID
 * @access  Private
 */
router.get('/:id', protect, async (req, res) => {
  try {
    const reminder = await Reminder.getById(req.params.id, req.user.id);
    res.json(reminder);
  } catch (error: any) {
    logger.error(`Get reminder error: ${error.message}`);
    if (error.message === 'Reminder not found') {
      return res.status(404).json({ message: error.message });
    }
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * @route   PUT /api/reminders/:id
 * @desc    Update a reminder
 * @access  Private
 */
router.put(
  '/:id',
  [
    protect,
    body('title').optional(),
    body('description').optional(),
    body('dueDate').optional().isISO8601().withMessage('Invalid date format'),
    validateRequest,
  ],
  async (req, res) => {
    try {
      const reminder = await Reminder.update(req.params.id, req.user.id, req.body);
      res.json(reminder);
    } catch (error: any) {
      logger.error(`Update reminder error: ${error.message}`);
      if (error.message === 'Reminder not found') {
        return res.status(404).json({ message: error.message });
      }
      res.status(500).json({ message: 'Server error' });
    }
  }
);

/**
 * @route   PUT /api/reminders/:id/complete
 * @desc    Mark a reminder as completed
 * @access  Private
 */
router.put('/:id/complete', protect, async (req, res) => {
  try {
    const completed = req.body.completed !== false; // Default to true if not specified
    const reminder = await Reminder.complete(req.params.id, req.user.id, completed);
    res.json(reminder);
  } catch (error: any) {
    logger.error(`Complete reminder error: ${error.message}`);
    if (error.message === 'Reminder not found') {
      return res.status(404).json({ message: error.message });
    }
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * @route   DELETE /api/reminders/:id
 * @desc    Delete a reminder
 * @access  Private
 */
router.delete('/:id', protect, async (req, res) => {
  try {
    const result = await Reminder.delete(req.params.id, req.user.id);
    res.json(result);
  } catch (error: any) {
    logger.error(`Delete reminder error: ${error.message}`);
    if (error.message === 'Reminder not found') {
      return res.status(404).json({ message: error.message });
    }
    res.status(500).json({ message: 'Server error' });
  }
});

export default router;