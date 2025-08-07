#!/bin/bash

# AstroAssistance Setup Script

echo "Setting up AstroAssistance project..."

# Create logs directory if it doesn't exist
mkdir -p /home/notcool/Desktop/AstroAssistance/logs

# Backend setup
echo "Setting up backend..."
cd /home/notcool/Desktop/AstroAssistance/backend

# Install dependencies
echo "Installing backend dependencies..."
npm install

# Create directories if they don't exist
mkdir -p /home/notcool/Desktop/AstroAssistance/backend/logs

# Setup Prisma and database
echo "Setting up database..."
npx prisma generate
npx prisma migrate dev --name init
npx prisma db seed

# Build TypeScript
echo "Building backend..."
npm run build

# Frontend setup
echo "Setting up frontend..."
cd /home/notcool/Desktop/AstroAssistance/frontend

# Install dependencies
echo "Installing frontend dependencies..."
npm install

# Build frontend
echo "Building frontend..."
npm run build

echo "Setup complete!"
echo "To start the backend server: cd /home/notcool/Desktop/AstroAssistance/backend && npm start"
echo "To start the frontend development server: cd /home/notcool/Desktop/AstroAssistance/frontend && npm run dev"
echo "Default login credentials: demo@example.com / password123"