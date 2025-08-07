import prisma from '../utils/prisma';
import { logger } from '../utils/logger';

export interface RecommendationInput {
  type: string;
  content: string;
  reason?: string;
}

export class Recommendation {
  /**
   * Create a new recommendation
   */
  static async create(userId: string, recommendationData: RecommendationInput) {
    try {
      const recommendation = await prisma.recommendation.create({
        data: {
          ...recommendationData,
          userId,
        },
      });

      return recommendation;
    } catch (error: any) {
      logger.error(`Create recommendation error: ${error.message}`);
      throw error;
    }
  }

  /**
   * Get all recommendations for a user
   */
  static async getAll(userId: string, filters: any = {}) {
    try {
      const where: any = { userId };

      // Apply filters
      if (filters.applied !== undefined) {
        where.applied = filters.applied === 'true';
      }

      if (filters.type) {
        where.type = filters.type;
      }

      // Get recommendations with pagination
      const page = parseInt(filters.page) || 1;
      const limit = parseInt(filters.limit) || 10;
      const skip = (page - 1) * limit;

      const [recommendations, total] = await Promise.all([
        prisma.recommendation.findMany({
          where,
          orderBy: { createdAt: 'desc' },
          skip,
          take: limit,
          include: {
            feedbacks: true,
          },
        }),
        prisma.recommendation.count({ where }),
      ]);

      return {
        recommendations,
        pagination: {
          page,
          limit,
          total,
          pages: Math.ceil(total / limit),
        },
      };
    } catch (error: any) {
      logger.error(`Get recommendations error: ${error.message}`);
      throw error;
    }
  }

  /**
   * Get a recommendation by ID
   */
  static async getById(id: string, userId: string) {
    try {
      const recommendation = await prisma.recommendation.findFirst({
        where: { id, userId },
        include: {
          feedbacks: true,
        },
      });

      if (!recommendation) {
        throw new Error('Recommendation not found');
      }

      return recommendation;
    } catch (error: any) {
      logger.error(`Get recommendation error: ${error.message}`);
      throw error;
    }
  }

  /**
   * Mark a recommendation as applied
   */
  static async apply(id: string, userId: string, applied: boolean = true) {
    try {
      // Check if recommendation exists and belongs to user
      const existingRecommendation = await prisma.recommendation.findFirst({
        where: { id, userId },
      });

      if (!existingRecommendation) {
        throw new Error('Recommendation not found');
      }

      const recommendation = await prisma.recommendation.update({
        where: { id },
        data: {
          applied,
          appliedAt: applied ? new Date() : null,
        },
      });

      return recommendation;
    } catch (error: any) {
      logger.error(`Apply recommendation error: ${error.message}`);
      throw error;
    }
  }

  /**
   * Generate recommendations for a user using AI models
   */
  static async generate(userId: string, count: number = 3) {
    try {
      // Import the AI service
      const { AIService } = require('../services/ai.service');
      
      logger.info(`Generating AI-powered recommendations for user: ${userId}`);
      
      // Get user data to base recommendations on
      const [tasks, goals, preferences] = await Promise.all([
        prisma.task.findMany({
          where: { userId, completed: false },
          orderBy: { dueDate: 'asc' },
        }),
        prisma.goal.findMany({
          where: { userId, completed: false },
        }),
        prisma.userPreference.findUnique({
          where: { userId },
        }),
      ]);

      // Use the AI service to generate recommendations
      const aiRecommendations = await AIService.generateRecommendations(userId, count);
      
      // Convert AI recommendations to database format
      const recommendationInputs: RecommendationInput[] = aiRecommendations.map(rec => ({
        type: rec.type,
        content: rec.content,
        reason: rec.reason,
      }));
      
      // Save recommendations to database
      const createdRecommendations = await Promise.all(
        recommendationInputs.map(rec => this.create(userId, rec))
      );

      return createdRecommendations;
    } catch (error: any) {
      logger.error(`Generate recommendations error: ${error.message}`);
      
      // Fallback to basic recommendations if AI fails
      logger.info('Falling back to basic recommendation generation');
      
      // Get user data to base recommendations on
      const [tasks, goals, preferences] = await Promise.all([
        prisma.task.findMany({
          where: { userId, completed: false },
          orderBy: { dueDate: 'asc' },
          take: 10,
        }),
        prisma.goal.findMany({
          where: { userId, completed: false },
          take: 5,
        }),
        prisma.userPreference.findUnique({
          where: { userId },
        }),
      ]);

      // Simple recommendation generation logic (fallback)
      const recommendations: RecommendationInput[] = [];

      // Task-based recommendations
      if (tasks.length > 0) {
        const highPriorityTasks = tasks.filter(task => task.priority === 'HIGH' || task.priority === 'high');
        if (highPriorityTasks.length > 0) {
          recommendations.push({
            type: 'task',
            content: `Focus on completing "${highPriorityTasks[0].title}" as it's a high priority task`,
            reason: 'High priority tasks should be completed first',
          });
        }

        const upcomingTasks = tasks.filter(task => task.dueDate && new Date(task.dueDate) < new Date(Date.now() + 86400000 * 2));
        if (upcomingTasks.length > 0) {
          recommendations.push({
            type: 'task',
            content: `"${upcomingTasks[0].title}" is due soon, consider working on it today`,
            reason: 'Tasks with approaching deadlines need attention',
          });
        }
      }

      // Goal-based recommendations
      if (goals.length > 0) {
        const lowProgressGoals = goals.filter(goal => goal.progress < 0.3);
        if (lowProgressGoals.length > 0) {
          recommendations.push({
            type: 'goal',
            content: `Make progress on your "${lowProgressGoals[0].title}" goal which is currently at ${Math.round(lowProgressGoals[0].progress * 100)}%`,
            reason: 'Regular progress on goals leads to successful completion',
          });
        }
      }

      // Break recommendations
      if (preferences?.breakReminders) {
        recommendations.push({
          type: 'break',
          content: 'Take a short break to refresh your mind',
          reason: 'Regular breaks improve productivity and focus',
        });
      }

      // Ensure we have at least the requested number of recommendations
      while (recommendations.length < count) {
        recommendations.push({
          type: 'productivity',
          content: 'Consider organizing your tasks by priority and deadline',
          reason: 'Organized task lists improve productivity',
        });
      }

      // Take only the requested number of recommendations
      const selectedRecommendations = recommendations.slice(0, count);

      // Save recommendations to database
      const createdRecommendations = await Promise.all(
        selectedRecommendations.map(rec => this.create(userId, rec))
      );

      return createdRecommendations;
    }
  }

  /**
   * Delete a recommendation
   */
  static async delete(id: string, userId: string) {
    try {
      // Check if recommendation exists and belongs to user
      const existingRecommendation = await prisma.recommendation.findFirst({
        where: { id, userId },
      });

      if (!existingRecommendation) {
        throw new Error('Recommendation not found');
      }

      await prisma.recommendation.delete({
        where: { id },
      });

      return { message: 'Recommendation deleted successfully' };
    } catch (error: any) {
      logger.error(`Delete recommendation error: ${error.message}`);
      throw error;
    }
  }
}