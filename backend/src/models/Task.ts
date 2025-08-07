import prisma from '../utils/prisma';
import { logger } from '../utils/logger';

export interface TaskInput {
  title: string;
  description?: string;
  category: string;
  priority: string;
  dueDate?: Date;
  estimatedDuration?: number;
  tags?: string[];
}

export class Task {
  /**
   * Create a new task with AI predictions
   */
  static async create(userId: string, taskData: TaskInput) {
    try {
      // Format the dueDate properly if it exists but isn't in ISO format
      let formattedData = { ...taskData };
      
      if (formattedData.dueDate && !(formattedData.dueDate instanceof Date) && !formattedData.dueDate.includes('T')) {
        // Convert YYYY-MM-DD to YYYY-MM-DDT00:00:00Z (ISO format)
        formattedData.dueDate = new Date(`${formattedData.dueDate}T00:00:00Z`);
      }
      
      // Create the task in the database
      const task = await prisma.task.create({
        data: {
          ...formattedData,
          tags: formattedData.tags || [],
          userId,
        },
      });
      
      try {
        // Import the AI service
        const { AIService } = require('../services/ai.service');
        
        // Get AI predictions for the task
        const predictions = await AIService.predictTaskCompletion(task);
        
        // Log the predictions (in a real app, you might store these or use them)
        logger.info(`AI predictions for task ${task.id}: ${JSON.stringify(predictions)}`);
        
        // If the task is predicted to not be completed on time with high confidence,
        // automatically generate a recommendation
        if (predictions && 
            predictions.completion_prediction && 
            !predictions.completion_prediction.will_complete_on_time && 
            predictions.completion_prediction.confidence > 0.7) {
          
          // Create a recommendation
          await prisma.recommendation.create({
            data: {
              type: 'task_completion_risk',
              content: `Your task "${task.title}" might be at risk of not being completed on time.`,
              reason: 'Based on AI analysis of your past task completion patterns',
              userId: userId,
            },
          });
          
          logger.info(`Created completion risk recommendation for task ${task.id}`);
        }
        
        // If the task is predicted to take significantly longer than estimated
        if (predictions && 
            predictions.duration_prediction && 
            predictions.duration_prediction.predicted_duration > 
            predictions.duration_prediction.estimated_duration * 1.5) {
          
          // Create a recommendation
          await prisma.recommendation.create({
            data: {
              type: 'task_duration_warning',
              content: `Your task "${task.title}" might take longer than estimated.`,
              reason: `Based on AI analysis, it might take about ${Math.round(predictions.duration_prediction.predicted_duration)} minutes instead of ${Math.round(predictions.duration_prediction.estimated_duration)} minutes.`,
              userId: userId,
            },
          });
          
          logger.info(`Created duration warning recommendation for task ${task.id}`);
        }
      } catch (aiError) {
        // If AI prediction fails, just log the error but don't fail the task creation
        logger.error(`AI prediction error for task ${task.id}: ${aiError.message}`);
      }

      return task;
    } catch (error: any) {
      logger.error(`Create task error: ${error.message}`);
      throw error;
    }
  }

  /**
   * Get all tasks for a user
   */
  static async getAll(userId: string, filters: any = {}) {
    try {
      const where: any = { userId };

      // Apply filters
      if (filters.completed !== undefined) {
        where.completed = filters.completed === 'true';
      }

      if (filters.category) {
        where.category = filters.category;
      }

      if (filters.priority) {
        where.priority = filters.priority;
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

      // Get tasks with pagination
      const page = parseInt(filters.page) || 1;
      const limit = parseInt(filters.limit) || 10;
      const skip = (page - 1) * limit;

      const [tasks, total] = await Promise.all([
        prisma.task.findMany({
          where,
          orderBy: { createdAt: 'desc' },
          skip,
          take: limit,
        }),
        prisma.task.count({ where }),
      ]);

      return {
        tasks,
        pagination: {
          page,
          limit,
          total,
          pages: Math.ceil(total / limit),
        },
      };
    } catch (error: any) {
      logger.error(`Get tasks error: ${error.message}`);
      throw error;
    }
  }

  /**
   * Get a task by ID
   */
  static async getById(id: string, userId: string) {
    try {
      const task = await prisma.task.findFirst({
        where: { id, userId },
      });

      if (!task) {
        throw new Error('Task not found');
      }

      return task;
    } catch (error: any) {
      logger.error(`Get task error: ${error.message}`);
      throw error;
    }
  }

  /**
   * Update a task
   */
  static async update(id: string, userId: string, taskData: Partial<TaskInput>) {
    try {
      // Check if task exists and belongs to user
      const existingTask = await prisma.task.findFirst({
        where: { id, userId },
      });

      if (!existingTask) {
        throw new Error('Task not found');
      }
      
      // Format the dueDate properly if it exists but isn't in ISO format
      let formattedData = { ...taskData };
      
      if (formattedData.dueDate && !(formattedData.dueDate instanceof Date) && !formattedData.dueDate.includes('T')) {
        // Convert YYYY-MM-DD to YYYY-MM-DDT00:00:00Z (ISO format)
        formattedData.dueDate = new Date(`${formattedData.dueDate}T00:00:00Z`);
      }

      const task = await prisma.task.update({
        where: { id },
        data: formattedData,
      });

      return task;
    } catch (error: any) {
      logger.error(`Update task error: ${error.message}`);
      throw error;
    }
  }

  /**
   * Mark a task as completed
   */
  static async complete(id: string, userId: string, completed: boolean = true) {
    try {
      // Check if task exists and belongs to user
      const existingTask = await prisma.task.findFirst({
        where: { id, userId },
      });

      if (!existingTask) {
        throw new Error('Task not found');
      }

      const task = await prisma.task.update({
        where: { id },
        data: {
          completed,
          completedAt: completed ? new Date() : null,
        },
      });

      return task;
    } catch (error: any) {
      logger.error(`Complete task error: ${error.message}`);
      throw error;
    }
  }

  /**
   * Delete a task
   */
  static async delete(id: string, userId: string) {
    try {
      // Check if task exists and belongs to user
      const existingTask = await prisma.task.findFirst({
        where: { id, userId },
      });

      if (!existingTask) {
        throw new Error('Task not found');
      }

      await prisma.task.delete({
        where: { id },
      });

      return { message: 'Task deleted successfully' };
    } catch (error: any) {
      logger.error(`Delete task error: ${error.message}`);
      throw error;
    }
  }
}