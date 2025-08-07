import prisma from '../utils/prisma';
import { logger } from '../utils/logger';

export interface LearningFeedbackInput {
  rating: number;
  comment?: string;
  recommendationId?: string;
}

export class LearningFeedback {
  /**
   * Create new feedback
   */
  static async create(userId: string, feedbackData: LearningFeedbackInput) {
    try {
      // Validate rating (1-5)
      if (feedbackData.rating < 1 || feedbackData.rating > 5) {
        throw new Error('Rating must be between 1 and 5');
      }

      // If recommendationId is provided, check if it exists and belongs to user
      if (feedbackData.recommendationId) {
        const recommendation = await prisma.recommendation.findFirst({
          where: {
            id: feedbackData.recommendationId,
            userId,
          },
        });

        if (!recommendation) {
          throw new Error('Recommendation not found');
        }
      }

      const feedback = await prisma.learningFeedback.create({
        data: {
          ...feedbackData,
          userId,
        },
      });

      return feedback;
    } catch (error: any) {
      logger.error(`Create feedback error: ${error.message}`);
      throw error;
    }
  }

  /**
   * Get all feedback for a user
   */
  static async getAll(userId: string, filters: any = {}) {
    try {
      const where: any = { userId };

      // Apply filters
      if (filters.recommendationId) {
        where.recommendationId = filters.recommendationId;
      }

      // Get feedback with pagination
      const page = parseInt(filters.page) || 1;
      const limit = parseInt(filters.limit) || 10;
      const skip = (page - 1) * limit;

      const [feedbacks, total] = await Promise.all([
        prisma.learningFeedback.findMany({
          where,
          orderBy: { createdAt: 'desc' },
          skip,
          take: limit,
          include: {
            recommendation: true,
          },
        }),
        prisma.learningFeedback.count({ where }),
      ]);

      return {
        feedbacks,
        pagination: {
          page,
          limit,
          total,
          pages: Math.ceil(total / limit),
        },
      };
    } catch (error: any) {
      logger.error(`Get feedbacks error: ${error.message}`);
      throw error;
    }
  }

  /**
   * Get feedback by ID
   */
  static async getById(id: string, userId: string) {
    try {
      const feedback = await prisma.learningFeedback.findFirst({
        where: { id, userId },
        include: {
          recommendation: true,
        },
      });

      if (!feedback) {
        throw new Error('Feedback not found');
      }

      return feedback;
    } catch (error: any) {
      logger.error(`Get feedback error: ${error.message}`);
      throw error;
    }
  }

  /**
   * Delete feedback
   */
  static async delete(id: string, userId: string) {
    try {
      // Check if feedback exists and belongs to user
      const existingFeedback = await prisma.learningFeedback.findFirst({
        where: { id, userId },
      });

      if (!existingFeedback) {
        throw new Error('Feedback not found');
      }

      await prisma.learningFeedback.delete({
        where: { id },
      });

      return { message: 'Feedback deleted successfully' };
    } catch (error: any) {
      logger.error(`Delete feedback error: ${error.message}`);
      throw error;
    }
  }
}