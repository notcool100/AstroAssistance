import prisma from '../utils/prisma';
import { logger } from '../utils/logger';

export interface UserPreferenceInput {
  workStartTime?: string;
  workEndTime?: string;
  focusHours?: string[];
  breakReminders?: boolean;
  breakInterval?: number;
  notificationEnabled?: boolean;
  theme?: string;
}

export class UserPreference {
  /**
   * Get user preferences
   */
  static async get(userId: string) {
    try {
      const preferences = await prisma.userPreference.findUnique({
        where: { userId },
      });

      if (!preferences) {
        // Create default preferences if not found
        return this.create(userId, {
          workStartTime: '09:00',
          workEndTime: '17:00',
          focusHours: [],
          breakReminders: true,
          breakInterval: 60,
          notificationEnabled: true,
          theme: 'light',
        });
      }

      return preferences;
    } catch (error: any) {
      logger.error(`Get user preferences error: ${error.message}`);
      throw error;
    }
  }

  /**
   * Create user preferences
   */
  static async create(userId: string, preferencesData: UserPreferenceInput) {
    try {
      // Check if preferences already exist
      const existingPreferences = await prisma.userPreference.findUnique({
        where: { userId },
      });

      if (existingPreferences) {
        return this.update(userId, preferencesData);
      }

      const preferences = await prisma.userPreference.create({
        data: {
          ...preferencesData,
          userId,
        },
      });

      return preferences;
    } catch (error: any) {
      logger.error(`Create user preferences error: ${error.message}`);
      throw error;
    }
  }

  /**
   * Update user preferences
   */
  static async update(userId: string, preferencesData: UserPreferenceInput) {
    try {
      // Check if preferences exist
      const existingPreferences = await prisma.userPreference.findUnique({
        where: { userId },
      });

      if (!existingPreferences) {
        return this.create(userId, preferencesData);
      }

      const preferences = await prisma.userPreference.update({
        where: { userId },
        data: preferencesData,
      });

      return preferences;
    } catch (error: any) {
      logger.error(`Update user preferences error: ${error.message}`);
      throw error;
    }
  }
}