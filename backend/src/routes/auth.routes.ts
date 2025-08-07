import express from 'express';
import { body } from 'express-validator';
import { User } from '../models';
import { protect } from '../middleware/auth';
import { validateRequest } from '../middleware/validate';
import { logger } from '../utils/logger';

const router = express.Router();

/**
 * @route   POST /api/auth/register
 * @desc    Register a new user
 * @access  Public
 */
router.post(
  '/register',
  [
    body('name').notEmpty().withMessage('Name is required'),
    body('email').isEmail().withMessage('Please include a valid email'),
    body('password')
      .isLength({ min: 6 })
      .withMessage('Password must be at least 6 characters'),
    validateRequest,
  ],
  async (req, res) => {
    try {
      const { name, email, password } = req.body;
      const user = await User.register({ name, email, password });
      res.status(201).json(user);
    } catch (error: any) {
      logger.error(`Register error: ${error.message}`);
      if (error.message === 'User already exists') {
        return res.status(400).json({ message: error.message });
      }
      res.status(500).json({ message: 'Server error' });
    }
  }
);

/**
 * @route   POST /api/auth/login
 * @desc    Authenticate user & get token
 * @access  Public
 */
router.post(
  '/login',
  [
    body('email').isEmail().withMessage('Please include a valid email'),
    body('password').exists().withMessage('Password is required'),
    validateRequest,
  ],
  async (req, res) => {
    try {
      const { email, password } = req.body;
      const user = await User.login({ email, password });
      res.json(user);
    } catch (error: any) {
      logger.error(`Login error: ${error.message}`);
      if (error.message === 'Invalid credentials') {
        return res.status(401).json({ message: error.message });
      }
      res.status(500).json({ message: 'Server error' });
    }
  }
);

/**
 * @route   GET /api/auth/me
 * @desc    Get current user
 * @access  Private
 */
router.get('/me', protect, async (req, res) => {
  try {
    const user = await User.getById(req.user.id);
    res.json(user);
  } catch (error: any) {
    logger.error(`Get user error: ${error.message}`);
    res.status(500).json({ message: 'Server error' });
  }
});

/**
 * @route   PUT /api/auth/me
 * @desc    Update user profile
 * @access  Private
 */
router.put(
  '/me',
  [
    protect,
    body('name').optional(),
    body('email').optional().isEmail().withMessage('Please include a valid email'),
    body('password')
      .optional()
      .isLength({ min: 6 })
      .withMessage('Password must be at least 6 characters'),
    validateRequest,
  ],
  async (req, res) => {
    try {
      const { name, email, password } = req.body;
      const user = await User.update(req.user.id, { name, email, password });
      res.json(user);
    } catch (error: any) {
      logger.error(`Update user error: ${error.message}`);
      res.status(500).json({ message: 'Server error' });
    }
  }
);

export default router;