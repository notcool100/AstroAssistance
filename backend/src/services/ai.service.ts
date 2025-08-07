/**
 * AI Service
 * 
 * This service handles the integration with the AI models for task prediction
 * and recommendation generation.
 */
import path from 'path';
import fs from 'fs';
import { spawn } from 'child_process';
import prisma from '../utils/prisma';
import { logger } from '../utils/logger';

// Path to the AI models directory
const MODELS_DIR = path.join(__dirname, '..', '..', '..', 'models');
// Path to the Python scripts
const SCRIPTS_DIR = path.join(__dirname, '..', '..', '..', 'ai');

// Ensure directories exist
if (!fs.existsSync(MODELS_DIR)) {
  fs.mkdirSync(MODELS_DIR, { recursive: true });
}
if (!fs.existsSync(SCRIPTS_DIR)) {
  fs.mkdirSync(SCRIPTS_DIR, { recursive: true });
}

export class AIService {
  /**
   * Predict if a task will be completed on time
   */
  static async predictTaskCompletion(taskData: any): Promise<any> {
    try {
      logger.info(`Predicting task completion for task: ${taskData.id}`);
      
      // Execute Python script for prediction
      return new Promise((resolve, reject) => {
        const pythonProcess = spawn('python', [
          path.join(SCRIPTS_DIR, 'predict_task.py'),
          '--task', JSON.stringify(taskData)
        ]);
        
        let result = '';
        let error = '';
        
        pythonProcess.stdout.on('data', (data) => {
          result += data.toString();
        });
        
        pythonProcess.stderr.on('data', (data) => {
          error += data.toString();
        });
        
        pythonProcess.on('close', (code) => {
          if (code !== 0) {
            logger.error(`Task prediction failed with code ${code}: ${error}`);
            reject(new Error(`Task prediction failed: ${error}`));
          } else {
            try {
              const prediction = JSON.parse(result);
              resolve(prediction);
            } catch (err) {
              reject(new Error(`Failed to parse prediction result: ${err.message}`));
            }
          }
        });
      });
    } catch (error: any) {
      logger.error(`Task prediction error: ${error.message}`);
      throw error;
    }
  }
  
  /**
   * Generate AI-powered recommendations for a user
   */
  static async generateRecommendations(userId: string, count: number = 3): Promise<any[]> {
    try {
      logger.info(`Generating AI recommendations for user: ${userId}`);
      
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
      
      // Execute Python script for recommendation generation
      return new Promise((resolve, reject) => {
        const pythonProcess = spawn('python', [
          path.join(SCRIPTS_DIR, 'generate_recommendations.py'),
          '--user', userId,
          '--count', count.toString(),
          '--tasks', JSON.stringify(tasks),
          '--goals', JSON.stringify(goals),
          '--preferences', JSON.stringify(preferences || {})
        ]);
        
        let result = '';
        let error = '';
        
        pythonProcess.stdout.on('data', (data) => {
          result += data.toString();
        });
        
        pythonProcess.stderr.on('data', (data) => {
          error += data.toString();
        });
        
        pythonProcess.on('close', (code) => {
          if (code !== 0) {
            logger.error(`Recommendation generation failed with code ${code}: ${error}`);
            // Fallback to basic recommendations if AI fails
            resolve(this.generateBasicRecommendations(userId, tasks, goals, preferences, count));
          } else {
            try {
              const recommendations = JSON.parse(result);
              resolve(recommendations);
            } catch (err) {
              logger.error(`Failed to parse recommendations: ${err.message}`);
              // Fallback to basic recommendations
              resolve(this.generateBasicRecommendations(userId, tasks, goals, preferences, count));
            }
          }
        });
      });
    } catch (error: any) {
      logger.error(`Recommendation generation error: ${error.message}`);
      throw error;
    }
  }
  
  /**
   * Generate basic recommendations without AI (fallback)
   */
  private static generateBasicRecommendations(
    userId: string, 
    tasks: any[], 
    goals: any[], 
    preferences: any,
    count: number
  ): any[] {
    logger.info(`Generating basic recommendations for user: ${userId}`);
    
    const recommendations = [];
    
    // Task-based recommendations
    if (tasks.length > 0) {
      const highPriorityTasks = tasks.filter(task => task.priority === 'HIGH');
      if (highPriorityTasks.length > 0) {
        recommendations.push({
          type: 'task',
          content: `Focus on completing "${highPriorityTasks[0].title}" as it's a high priority task`,
          reason: 'High priority tasks should be completed first',
        });
      }
      
      const upcomingTasks = tasks.filter(task => 
        task.dueDate && new Date(task.dueDate) < new Date(Date.now() + 86400000 * 2)
      );
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
    return recommendations.slice(0, count);
  }
  
  /**
   * Train the AI models with the latest data
   */
  static async trainModels(): Promise<boolean> {
    try {
      logger.info('Starting AI model training');
      
      return new Promise((resolve, reject) => {
        const pythonProcess = spawn('python', [
          path.join(SCRIPTS_DIR, 'train_models.py')
        ]);
        
        let result = '';
        let error = '';
        
        pythonProcess.stdout.on('data', (data) => {
          result += data.toString();
        });
        
        pythonProcess.stderr.on('data', (data) => {
          error += data.toString();
        });
        
        pythonProcess.on('close', (code) => {
          if (code !== 0) {
            logger.error(`Model training failed with code ${code}: ${error}`);
            reject(new Error(`Model training failed: ${error}`));
          } else {
            logger.info(`Model training completed: ${result}`);
            resolve(true);
          }
        });
      });
    } catch (error: any) {
      logger.error(`Model training error: ${error.message}`);
      throw error;
    }
  }
  
  /**
   * Process user feedback on recommendations
   */
  static async processFeedback(feedbackData: any): Promise<void> {
    try {
      logger.info(`Processing feedback for recommendation: ${feedbackData.recommendationId}`);
      
      // Store feedback for future model training
      await prisma.learningFeedback.create({
        data: {
          rating: feedbackData.rating,
          comment: feedbackData.comment,
          userId: feedbackData.userId,
          recommendationId: feedbackData.recommendationId,
        },
      });
      
      // This feedback will be used in the next training cycle
    } catch (error: any) {
      logger.error(`Process feedback error: ${error.message}`);
      throw error;
    }
  }
}