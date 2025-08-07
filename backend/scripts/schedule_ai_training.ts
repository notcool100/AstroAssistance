/**
 * AI Training Scheduler
 * 
 * This script schedules regular AI training using node-cron.
 * It exports data from the database and then triggers the AI training process.
 */
import { spawn } from 'child_process';
import path from 'path';
import cron from 'node-cron';
import dotenv from 'dotenv';
import { logger } from '../src/utils/logger';

// Load environment variables
dotenv.config();

// Paths
const SCRIPTS_DIR = path.join(__dirname);
const AI_DIR = path.join(__dirname, '..', 'ai');

// Function to export data for training
async function exportData(): Promise<boolean> {
  return new Promise((resolve, reject) => {
    logger.info('Exporting data for AI training...');
    
    const exportProcess = spawn('npx', ['ts-node', path.join(SCRIPTS_DIR, 'export_data_for_training.ts')]);
    
    exportProcess.stdout.on('data', (data) => {
      logger.info(`Export: ${data.toString().trim()}`);
    });
    
    exportProcess.stderr.on('data', (data) => {
      logger.error(`Export error: ${data.toString().trim()}`);
    });
    
    exportProcess.on('close', (code) => {
      if (code === 0) {
        logger.info('Data export completed successfully');
        resolve(true);
      } else {
        logger.error(`Data export failed with code ${code}`);
        resolve(false);
      }
    });
  });
}

// Function to train AI models
async function trainModels(): Promise<boolean> {
  return new Promise((resolve, reject) => {
    logger.info('Training AI models...');
    
    const trainProcess = spawn('python', [path.join(AI_DIR, 'train_models.py')]);
    
    trainProcess.stdout.on('data', (data) => {
      logger.info(`Training: ${data.toString().trim()}`);
    });
    
    trainProcess.stderr.on('data', (data) => {
      logger.error(`Training error: ${data.toString().trim()}`);
    });
    
    trainProcess.on('close', (code) => {
      if (code === 0) {
        logger.info('AI training completed successfully');
        resolve(true);
      } else {
        logger.error(`AI training failed with code ${code}`);
        resolve(false);
      }
    });
  });
}

// Main function to run the training process
async function runTraining() {
  logger.info('Starting scheduled AI training process');
  
  try {
    // Export data
    const exportSuccess = await exportData();
    
    if (!exportSuccess) {
      logger.error('Skipping AI training due to data export failure');
      return;
    }
    
    // Train models
    const trainingSuccess = await trainModels();
    
    if (trainingSuccess) {
      logger.info('Scheduled AI training process completed successfully');
    } else {
      logger.error('Scheduled AI training process failed');
    }
  } catch (error) {
    logger.error(`Error in scheduled AI training: ${error.message}`);
  }
}

// Schedule training to run daily at 2 AM
cron.schedule('0 2 * * *', () => {
  logger.info('Running scheduled AI training');
  runTraining();
});

// Also allow manual triggering
if (process.argv.includes('--run-now')) {
  logger.info('Manually triggering AI training');
  runTraining();
}

logger.info('AI training scheduler started');
logger.info('Training will run daily at 2 AM');

// Keep the process running
process.stdin.resume();