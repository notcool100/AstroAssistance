import { writable } from 'svelte/store';
import axios from 'axios';

interface Task {
  id: string;
  title: string;
  description?: string;
  category: string;
  priority: string;
  dueDate?: string;
  estimatedDuration?: number;
  tags: string[];
  completed: boolean;
  completedAt?: string;
  createdAt: string;
  updatedAt: string;
}

interface TasksState {
  tasks: Task[];
  currentTask: Task | null;
  loading: boolean;
  error: string | null;
  pagination: {
    page: number;
    limit: number;
    total: number;
    pages: number;
  };
}

const initialState: TasksState = {
  tasks: [],
  currentTask: null,
  loading: false,
  error: null,
  pagination: {
    page: 1,
    limit: 10,
    total: 0,
    pages: 0,
  },
};

function createTasksStore() {
  const { subscribe, set, update } = writable<TasksState>(initialState);

  return {
    subscribe,
    
    fetchTasks: async (filters: any = {}) => {
      update(state => ({ ...state, loading: true, error: null }));
      
      try {
        const queryParams = new URLSearchParams();
        
        // Add filters to query params
        Object.entries(filters).forEach(([key, value]) => {
          if (value !== undefined && value !== null && value !== '') {
            queryParams.append(key, String(value));
          }
        });
        
        const res = await axios.get(`/api/tasks?${queryParams.toString()}`);
        
        update(state => ({
          ...state,
          tasks: res.data.tasks,
          pagination: res.data.pagination,
          loading: false,
        }));
      } catch (err: any) {
        const errorMessage = err.response?.data?.message || 'Failed to fetch tasks';
        
        update(state => ({
          ...state,
          loading: false,
          error: errorMessage,
        }));
      }
    },
    
    fetchTask: async (id: string) => {
      update(state => ({ ...state, loading: true, error: null, currentTask: null }));
      
      try {
        const res = await axios.get(`/api/tasks/${id}`);
        
        update(state => ({
          ...state,
          currentTask: res.data,
          loading: false,
        }));
      } catch (err: any) {
        const errorMessage = err.response?.data?.message || 'Failed to fetch task';
        
        update(state => ({
          ...state,
          loading: false,
          error: errorMessage,
        }));
      }
    },
    
    createTask: async (taskData: Omit<Task, 'id' | 'completed' | 'completedAt' | 'createdAt' | 'updatedAt'>) => {
      update(state => ({ ...state, loading: true, error: null }));
      
      try {
        const res = await axios.post('/api/tasks', taskData);
        
        update(state => ({
          ...state,
          tasks: [res.data, ...state.tasks],
          loading: false,
        }));
        
        return res.data;
      } catch (err: any) {
        const errorMessage = err.response?.data?.message || 'Failed to create task';
        
        update(state => ({
          ...state,
          loading: false,
          error: errorMessage,
        }));
        
        return null;
      }
    },
    
    updateTask: async (id: string, taskData: Partial<Task>) => {
      update(state => ({ ...state, loading: true, error: null }));
      
      try {
        const res = await axios.put(`/api/tasks/${id}`, taskData);
        
        update(state => ({
          ...state,
          tasks: state.tasks.map(task => task.id === id ? res.data : task),
          currentTask: state.currentTask?.id === id ? res.data : state.currentTask,
          loading: false,
        }));
        
        return res.data;
      } catch (err: any) {
        const errorMessage = err.response?.data?.message || 'Failed to update task';
        
        update(state => ({
          ...state,
          loading: false,
          error: errorMessage,
        }));
        
        return null;
      }
    },
    
    completeTask: async (id: string, completed: boolean = true) => {
      update(state => ({ ...state, loading: true, error: null }));
      
      try {
        const res = await axios.put(`/api/tasks/${id}/complete`, { completed });
        
        update(state => ({
          ...state,
          tasks: state.tasks.map(task => task.id === id ? res.data : task),
          currentTask: state.currentTask?.id === id ? res.data : state.currentTask,
          loading: false,
        }));
        
        return res.data;
      } catch (err: any) {
        const errorMessage = err.response?.data?.message || 'Failed to update task status';
        
        update(state => ({
          ...state,
          loading: false,
          error: errorMessage,
        }));
        
        return null;
      }
    },
    
    deleteTask: async (id: string) => {
      update(state => ({ ...state, loading: true, error: null }));
      
      try {
        await axios.delete(`/api/tasks/${id}`);
        
        update(state => ({
          ...state,
          tasks: state.tasks.filter(task => task.id !== id),
          currentTask: state.currentTask?.id === id ? null : state.currentTask,
          loading: false,
        }));
        
        return true;
      } catch (err: any) {
        const errorMessage = err.response?.data?.message || 'Failed to delete task';
        
        update(state => ({
          ...state,
          loading: false,
          error: errorMessage,
        }));
        
        return false;
      }
    },
    
    clearError: () => {
      update(state => ({ ...state, error: null }));
    },
  };
}

export const tasksStore = createTasksStore();