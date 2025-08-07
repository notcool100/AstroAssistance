import express from 'express';
import { body } from 'express-validator';
import { UserPreference } from '../models';
import { protect } from '../middleware/auth';
import { validateRequest } from '../middleware/validate';
import { logger } from '../utils/logger';

const router = express.Router();

/**
 * @route   GET /api/preferences
 * @desc    Get user preferences
 * @access  Private
 */
router.get('/', protect, async (req, res) => {
  try {
    const preferences = await UserPreference.get(req.user.id);
    res.json(preferences);
  } catch (error: any) {
    logger.error(`Get preferences error: ${error.message}`);
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * @route   PUT /api/preferences
 * @desc    Update user preferences
 * @access  Private
 */
router.put(
  '/',
  [
    protect,
    body('workStartTime').optional().matches(/^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$/).withMessage('Work start time must be in format HH:MM'),
    body('workEndTime').optional().matches(/^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$/).withMessage('Work end time must be in format HH:MM'),
    body('focusHours').optional().isArray().withMessage('Focus hours must be an array'),
    body('breakReminders').optional().isBoolean().withMessage('Break reminders must be a boolean'),
    body('breakInterval').optional().isInt({ min: 15, max: 120 }).withMessage('Break interval must be between 15 and 120 minutes'),
    body('notificationEnabled').optional().isBoolean().withMessage('Notification enabled must be a boolean'),
    body('theme').optional().isIn(['light', 'dark', 'system']).withMessage('Theme must be light, dark, or system'),
    validateRequest,
  ],
  async (req, res) => {
    try {
      const preferences = await UserPreference.update(req.user.id, req.body);
      res.json(preferences);
    } catch (error: any) {
      logger.error(`Update preferences error: ${error.message}`);
      res.status(500).json({ message: 'Server error' });
    }
  }
);

export default router;