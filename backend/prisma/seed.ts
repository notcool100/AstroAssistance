import { PrismaClient } from '@prisma/client';
import bcrypt from 'bcryptjs';

const prisma = new PrismaClient();

async function main() {
  // Clear existing data
  await prisma.learningFeedback.deleteMany();
  await prisma.recommendation.deleteMany();
  await prisma.userPreference.deleteMany();
  await prisma.goal.deleteMany();
  await prisma.reminder.deleteMany();
  await prisma.task.deleteMany();
  await prisma.user.deleteMany();

  // Create demo user
  const demoUser = await prisma.user.create({
    data: {
      email: 'demo@example.com',
      password: await bcrypt.hash('password123', 10),
      name: 'Demo User',
    },
  });

  // Create user preferences
  await prisma.userPreference.create({
    data: {
      userId: demoUser.id,
      workStartTime: '09:00',
      workEndTime: '17:00',
      focusHours: ['10:00-12:00', '14:00-16:00'],
      breakReminders: true,
      breakInterval: 60,
      notificationEnabled: true,
      theme: 'light',
    },
  });

  // Create sample tasks
  const tasks = [
    {
      title: 'Complete project proposal',
      description: 'Finish the draft and send for review',
      category: 'work',
      priority: 'high',
      dueDate: new Date(Date.now() + 86400000 * 2), // 2 days from now
      estimatedDuration: 120,
      tags: ['project', 'proposal'],
      userId: demoUser.id,
    },
    {
      title: 'Weekly team meeting',
      description: 'Discuss project progress and next steps',
      category: 'work',
      priority: 'medium',
      dueDate: new Date(Date.now() + 86400000 * 1), // 1 day from now
      estimatedDuration: 60,
      tags: ['meeting', 'team'],
      userId: demoUser.id,
    },
    {
      title: 'Grocery shopping',
      description: 'Buy fruits, vegetables, and other essentials',
      category: 'personal',
      priority: 'low',
      dueDate: new Date(Date.now() + 86400000 * 3), // 3 days from now
      estimatedDuration: 45,
      tags: ['shopping', 'errands'],
      userId: demoUser.id,
    },
  ];

  for (const task of tasks) {
    await prisma.task.create({ data: task });
  }

  // Create sample reminders
  const reminders = [
    {
      title: 'Call dentist',
      description: 'Schedule annual checkup',
      dueDate: new Date(Date.now() + 86400000 * 1), // 1 day from now
      userId: demoUser.id,
    },
    {
      title: 'Pay electricity bill',
      description: 'Due by the end of the month',
      dueDate: new Date(Date.now() + 86400000 * 5), // 5 days from now
      userId: demoUser.id,
    },
  ];

  for (const reminder of reminders) {
    await prisma.reminder.create({ data: reminder });
  }

  // Create sample goals
  const goals = [
    {
      title: 'Learn TypeScript',
      description: 'Complete online course and build a project',
      targetDate: new Date(Date.now() + 86400000 * 30), // 30 days from now
      progress: 0.25,
      userId: demoUser.id,
    },
    {
      title: 'Exercise regularly',
      description: 'Go to the gym at least 3 times a week',
      targetDate: new Date(Date.now() + 86400000 * 90), // 90 days from now
      progress: 0.1,
      userId: demoUser.id,
    },
  ];

  for (const goal of goals) {
    await prisma.goal.create({ data: goal });
  }

  // Create sample recommendations
  const recommendations = [
    {
      type: 'task',
      content: 'Consider breaking down the project proposal into smaller tasks',
      reason: 'Large tasks are more manageable when divided into smaller components',
      userId: demoUser.id,
    },
    {
      type: 'break',
      content: 'Take a 15-minute break after your team meeting',
      reason: 'You have been working for 2 hours straight',
      userId: demoUser.id,
    },
    {
      type: 'goal',
      content: 'Set a specific target for your TypeScript learning goal',
      reason: 'Specific goals are more likely to be achieved',
      userId: demoUser.id,
    },
  ];

  for (const recommendation of recommendations) {
    await prisma.recommendation.create({ data: recommendation });
  }

  console.log('Database has been seeded!');
}

main()
  .catch((e) => {
    console.error(e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });