import prisma from '../utils/prisma';
import { logger } from '../utils/logger';

export interface ReminderInput {
  title: string;
  description?: string;
  dueDate: Date;
}

export class Reminder {
  /**
   * Create a new reminder
   */
  static async create(userId: string, reminderData: ReminderInput) {
    try {
      const reminder = await prisma.reminder.create({
        data: {
          ...reminderData,
          userId,
        },
      });

      return reminder;
    } catch (error: any) {
      logger.error(`Create reminder error: ${error.message}`);
      throw error;
    }
  }

  /**
   * Get all reminders for a user
   */
  static async getAll(userId: string, filters: any = {}) {
    try {
      const where: any = { userId };

      // Apply filters
      if (filters.completed !== undefined) {
        where.completed = filters.completed === 'true';
      }

      if (filters.dueDate) {
        const date = new Date(filters.dueDate);
        where.dueDate = {
          gte: new Date(date.setHours(0, 0, 0, 0)),
          lt: new Date(date.setHours(23, 59, 59, 999)),
        };
      }

      if (filters.search) {
        where.OR = [
          { title: { contains: filters.search, mode: 'insensitive' } },
          { description: { contains: filters.search, mode: 'insensitive' } },
        ];
      }

      // Get reminders with pagination
      const page = parseInt(filters.page) || 1;
      const limit = parseInt(filters.limit) || 10;
      const skip = (page - 1) * limit;

      const [reminders, total] = await Promise.all([
        prisma.reminder.findMany({
          where,
          orderBy: { dueDate: 'asc' },
          skip,
          take: limit,
        }),
        prisma.reminder.count({ where }),
      ]);

      return {
        reminders,
        pagination: {
          page,
          limit,
          total,
          pages: Math.ceil(total / limit),
        },
      };
    } catch (error: any) {
      logger.error(`Get reminders error: ${error.message}`);
      throw error;
    }
  }

  /**
   * Get a reminder by ID
   */
  static async getById(id: string, userId: string) {
    try {
      const reminder = await prisma.reminder.findFirst({
        where: { id, userId },
      });

      if (!reminder) {
        throw new Error('Reminder not found');
      }

      return reminder;
    } catch (error: any) {
      logger.error(`Get reminder error: ${error.message}`);
      throw error;
    }
  }

  /**
   * Update a reminder
   */
  static async update(id: string, userId: string, reminderData: Partial<ReminderInput>) {
    try {
      // Check if reminder exists and belongs to user
      const existingReminder = await prisma.reminder.findFirst({
        where: { id, userId },
      });

      if (!existingReminder) {
        throw new Error('Reminder not found');
      }

      const reminder = await prisma.reminder.update({
        where: { id },
        data: reminderData,
      });

      return reminder;
    } catch (error: any) {
      logger.error(`Update reminder error: ${error.message}`);
      throw error;
    }
  }

  /**
   * Mark a reminder as completed
   */
  static async complete(id: string, userId: string, completed: boolean = true) {
    try {
      // Check if reminder exists and belongs to user
      const existingReminder = await prisma.reminder.findFirst({
        where: { id, userId },
      });

      if (!existingReminder) {
        throw new Error('Reminder not found');
      }

      const reminder = await prisma.reminder.update({
        where: { id },
        data: { completed },
      });

      return reminder;
    } catch (error: any) {
      logger.error(`Complete reminder error: ${error.message}`);
      throw error;
    }
  }

  /**
   * Delete a reminder
   */
  static async delete(id: string, userId: string) {
    try {
      // Check if reminder exists and belongs to user
      const existingReminder = await prisma.reminder.findFirst({
        where: { id, userId },
      });

      if (!existingReminder) {
        throw new Error('Reminder not found');
      }

      await prisma.reminder.delete({
        where: { id },
      });

      return { message: 'Reminder deleted successfully' };
    } catch (error: any) {
      logger.error(`Delete reminder error: ${error.message}`);
      throw error;
    }
  }
}