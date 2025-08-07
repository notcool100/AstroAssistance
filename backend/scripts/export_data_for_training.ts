/**
 * Data Export Script for AI Training
 * 
 * This script exports data from the database to be used for training the AI models.
 * It exports tasks, goals, and user preferences to JSON files.
 */
import fs from 'fs';
import path from 'path';
import { PrismaClient } from '@prisma/client';
import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

// Initialize Prisma client
const prisma = new PrismaClient();

// Output directory
const DATA_DIR = path.join(__dirname, '..', '..', 'data');

// Ensure data directory exists
if (!fs.existsSync(DATA_DIR)) {
  fs.mkdirSync(DATA_DIR, { recursive: true });
}

async function exportTasks() {
  console.log('Exporting tasks...');
  
  // Get all tasks
  const tasks = await prisma.task.findMany({
    orderBy: { createdAt: 'desc' },
  });
  
  // Write to file
  fs.writeFileSync(
    path.join(DATA_DIR, 'tasks_export.json'),
    JSON.stringify(tasks, null, 2)
  );
  
  console.log(`Exported ${tasks.length} tasks`);
}

async function exportGoals() {
  console.log('Exporting goals...');
  
  // Get all goals
  const goals = await prisma.goal.findMany({
    orderBy: { createdAt: 'desc' },
  });
  
  // Write to file
  fs.writeFileSync(
    path.join(DATA_DIR, 'goals_export.json'),
    JSON.stringify(goals, null, 2)
  );
  
  console.log(`Exported ${goals.length} goals`);
}

async function exportUserPreferences() {
  console.log('Exporting user preferences...');
  
  // Get all user preferences
  const preferences = await prisma.userPreference.findMany();
  
  // Write to file
  fs.writeFileSync(
    path.join(DATA_DIR, 'preferences_export.json'),
    JSON.stringify(preferences, null, 2)
  );
  
  console.log(`Exported ${preferences.length} user preferences`);
}

async function exportRecommendationFeedback() {
  console.log('Exporting recommendation feedback...');
  
  // Get all feedback
  const feedback = await prisma.learningFeedback.findMany({
    include: {
      recommendation: true,
    },
  });
  
  // Write to file
  fs.writeFileSync(
    path.join(DATA_DIR, 'feedback_export.json'),
    JSON.stringify(feedback, null, 2)
  );
  
  console.log(`Exported ${feedback.length} feedback entries`);
}

async function main() {
  try {
    console.log('Starting data export for AI training...');
    
    // Export all data
    await exportTasks();
    await exportGoals();
    await exportUserPreferences();
    await exportRecommendationFeedback();
    
    console.log('Data export completed successfully');
  } catch (error) {
    console.error('Error exporting data:', error);
  } finally {
    await prisma.$disconnect();
  }
}

// Run the export
main();