import prisma from '../utils/prisma';
import { logger } from '../utils/logger';

export interface GoalInput {
  title: string;
  description?: string;
  targetDate?: Date;
  progress?: number;
}

export class Goal {
  /**
   * Create a new goal
   */
  static async create(userId: string, goalData: GoalInput) {
    try {
      const goal = await prisma.goal.create({
        data: {
          ...goalData,
          progress: goalData.progress || 0,
          userId,
        },
      });

      return goal;
    } catch (error: any) {
      logger.error(`Create goal error: ${error.message}`);
      throw error;
    }
  }

  /**
   * Get all goals for a user
   */
  static async getAll(userId: string, filters: any = {}) {
    try {
      const where: any = { userId };

      // Apply filters
      if (filters.completed !== undefined) {
        where.completed = filters.completed === 'true';
      }

      if (filters.search) {
        where.OR = [
          { title: { contains: filters.search, mode: 'insensitive' } },
          { description: { contains: filters.search, mode: 'insensitive' } },
        ];
      }

      // Get goals with pagination
      const page = parseInt(filters.page) || 1;
      const limit = parseInt(filters.limit) || 10;
      const skip = (page - 1) * limit;

      const [goals, total] = await Promise.all([
        prisma.goal.findMany({
          where,
          orderBy: { createdAt: 'desc' },
          skip,
          take: limit,
        }),
        prisma.goal.count({ where }),
      ]);

      return {
        goals,
        pagination: {
          page,
          limit,
          total,
          pages: Math.ceil(total / limit),
        },
      };
    } catch (error: any) {
      logger.error(`Get goals error: ${error.message}`);
      throw error;
    }
  }

  /**
   * Get a goal by ID
   */
  static async getById(id: string, userId: string) {
    try {
      const goal = await prisma.goal.findFirst({
        where: { id, userId },
      });

      if (!goal) {
        throw new Error('Goal not found');
      }

      return goal;
    } catch (error: any) {
      logger.error(`Get goal error: ${error.message}`);
      throw error;
    }
  }

  /**
   * Update a goal
   */
  static async update(id: string, userId: string, goalData: Partial<GoalInput>) {
    try {
      // Check if goal exists and belongs to user
      const existingGoal = await prisma.goal.findFirst({
        where: { id, userId },
      });

      if (!existingGoal) {
        throw new Error('Goal not found');
      }

      // If progress is 1 (100%), mark as completed
      let completed = existingGoal.completed;
      if (goalData.progress !== undefined) {
        if (goalData.progress >= 1) {
          completed = true;
        } else if (goalData.progress < 1 && existingGoal.completed) {
          completed = false;
        }
      }

      const goal = await prisma.goal.update({
        where: { id },
        data: {
          ...goalData,
          completed,
        },
      });

      return goal;
    } catch (error: any) {
      logger.error(`Update goal error: ${error.message}`);
      throw error;
    }
  }

  /**
   * Mark a goal as completed
   */
  static async complete(id: string, userId: string, completed: boolean = true) {
    try {
      // Check if goal exists and belongs to user
      const existingGoal = await prisma.goal.findFirst({
        where: { id, userId },
      });

      if (!existingGoal) {
        throw new Error('Goal not found');
      }

      const goal = await prisma.goal.update({
        where: { id },
        data: {
          completed,
          progress: completed ? 1 : existingGoal.progress,
        },
      });

      return goal;
    } catch (error: any) {
      logger.error(`Complete goal error: ${error.message}`);
      throw error;
    }
  }

  /**
   * Delete a goal
   */
  static async delete(id: string, userId: string) {
    try {
      // Check if goal exists and belongs to user
      const existingGoal = await prisma.goal.findFirst({
        where: { id, userId },
      });

      if (!existingGoal) {
        throw new Error('Goal not found');
      }

      await prisma.goal.delete({
        where: { id },
      });

      return { message: 'Goal deleted successfully' };
    } catch (error: any) {
      logger.error(`Delete goal error: ${error.message}`);
      throw error;
    }
  }
}