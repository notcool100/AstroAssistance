import prisma from '../utils/prisma';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import { logger } from '../utils/logger';

export interface UserInput {
  email: string;
  password: string;
  name: string;
}

export interface UserLoginInput {
  email: string;
  password: string;
}

export class User {
  /**
   * Register a new user
   */
  static async register(userData: UserInput) {
    try {
      // Check if user already exists
      const existingUser = await prisma.user.findUnique({
        where: { email: userData.email },
      });

      if (existingUser) {
        throw new Error('User already exists');
      }

      // Hash password
      const salt = await bcrypt.genSalt(10);
      const hashedPassword = await bcrypt.hash(userData.password, salt);

      // Create user
      const user = await prisma.user.create({
        data: {
          email: userData.email,
          password: hashedPassword,
          name: userData.name,
        },
      });

      // Create default user preferences
      await prisma.userPreference.create({
        data: {
          userId: user.id,
          workStartTime: '09:00',
          workEndTime: '17:00',
          focusHours: [],
          breakReminders: true,
          breakInterval: 60,
          notificationEnabled: true,
          theme: 'light',
        },
      });

      return {
        id: user.id,
        email: user.email,
        name: user.name,
        token: this.generateToken(user.id),
      };
    } catch (error: any) {
      logger.error(`User registration error: ${error.message}`);
      throw error;
    }
  }

  /**
   * Login a user
   */
  static async login(loginData: UserLoginInput) {
    try {
      // Find user by email
      const user = await prisma.user.findUnique({
        where: { email: loginData.email },
      });

      if (!user) {
        throw new Error('Invalid credentials');
      }

      // Check password
      const isMatch = await bcrypt.compare(loginData.password, user.password);

      if (!isMatch) {
        throw new Error('Invalid credentials');
      }

      return {
        id: user.id,
        email: user.email,
        name: user.name,
        token: this.generateToken(user.id),
      };
    } catch (error: any) {
      logger.error(`User login error: ${error.message}`);
      throw error;
    }
  }

  /**
   * Get user by ID
   */
  static async getById(id: string) {
    try {
      const user = await prisma.user.findUnique({
        where: { id },
        select: {
          id: true,
          email: true,
          name: true,
          createdAt: true,
          updatedAt: true,
        },
      });

      if (!user) {
        throw new Error('User not found');
      }

      return user;
    } catch (error: any) {
      logger.error(`Get user error: ${error.message}`);
      throw error;
    }
  }

  /**
   * Update user
   */
  static async update(id: string, userData: Partial<UserInput>) {
    try {
      // If password is provided, hash it
      if (userData.password) {
        const salt = await bcrypt.genSalt(10);
        userData.password = await bcrypt.hash(userData.password, salt);
      }

      const user = await prisma.user.update({
        where: { id },
        data: userData,
        select: {
          id: true,
          email: true,
          name: true,
          createdAt: true,
          updatedAt: true,
        },
      });

      return user;
    } catch (error: any) {
      logger.error(`Update user error: ${error.message}`);
      throw error;
    }
  }

  /**
   * Delete user
   */
  static async delete(id: string) {
    try {
      await prisma.user.delete({
        where: { id },
      });

      return { message: 'User deleted successfully' };
    } catch (error: any) {
      logger.error(`Delete user error: ${error.message}`);
      throw error;
    }
  }

  /**
   * Generate JWT token
   */
  private static generateToken(id: string) {
    return jwt.sign({ id }, process.env.JWT_SECRET as string, {
      expiresIn: process.env.JWT_EXPIRES_IN || '7d',
    });
  }
}