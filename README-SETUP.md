# AstroAssistance Setup Guide

This guide will help you set up and run the AstroAssistance project, which consists of a Node.js TypeScript backend with PostgreSQL database and a Svelte TypeScript frontend.

## Prerequisites

- Node.js (v16 or higher)
- PostgreSQL (v12 or higher)
- npm or yarn

## Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd /home/notcool/Desktop/AstroAssistance/backend
   ```

2. Install dependencies:
   ```bash
   npm install
   # or
   yarn install
   ```

3. Create a PostgreSQL database:
   ```bash
   createdb astro_assistance
   ```

4. Set up environment variables:
   - The `.env` file should already be created with default values
   - Update the `DATABASE_URL` in `.env` if your PostgreSQL configuration is different

5. Run database migrations:
   ```bash
   npx prisma migrate dev
   ```

6. Seed the database with initial data:
   ```bash
   npx prisma db seed
   ```

7. Build the TypeScript code:
   ```bash
   npm run build
   # or
   yarn build
   ```

8. Start the backend server:
   ```bash
   npm start
   # or
   yarn start
   ```

   For development with auto-reload:
   ```bash
   npm run dev
   # or
   yarn dev
   ```

The backend server will run on http://localhost:8000 by default.

## Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd /home/notcool/Desktop/AstroAssistance/frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   # or
   yarn install
   ```

3. Start the development server:
   ```bash
   npm run dev
   # or
   yarn dev
   ```

4. For production build:
   ```bash
   npm run build
   # or
   yarn build
   ```

The frontend development server will run on http://localhost:5173 by default.

## Default User Credentials

After seeding the database, you can log in with the following credentials:

- Email: demo@example.com
- Password: password123

## API Documentation

The backend API provides the following main endpoints:

- `/api/auth`: Authentication endpoints (register, login)
- `/api/tasks`: Task management
- `/api/reminders`: Reminder management
- `/api/goals`: Goal management
- `/api/preferences`: User preferences
- `/api/recommendations`: AI recommendations
- `/api/feedback`: Learning feedback

## Project Structure

### Backend

```
backend/
├── prisma/              # Database schema and migrations
├── src/                 # Source code
│   ├── middleware/      # Express middleware
│   ├── models/          # Data models
│   ├── routes/          # API routes
│   ├── utils/           # Utility functions
│   └── server.ts        # Main server file
├── .env                 # Environment variables
├── package.json         # Dependencies and scripts
└── tsconfig.json        # TypeScript configuration
```

### Frontend

```
frontend/
├── public/              # Static assets
├── src/                 # Source code
│   ├── components/      # Reusable components
│   ├── pages/           # Page components
│   ├── stores/          # Svelte stores
│   ├── App.svelte       # Main app component
│   └── main.ts          # Entry point
├── index.html           # HTML template
├── package.json         # Dependencies and scripts
└── tsconfig.json        # TypeScript configuration
```

## Technologies Used

### Backend
- Node.js with Express
- TypeScript
- Prisma ORM
- PostgreSQL
- JWT Authentication
- Winston Logger

### Frontend
- Svelte
- TypeScript
- TailwindCSS
- Chart.js
- Svelte Navigator
- Axios