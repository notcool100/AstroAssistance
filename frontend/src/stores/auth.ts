import { writable } from 'svelte/store';
import axios from 'axios';

interface AuthState {
  token: string | null;
  user: any | null;
  loading: boolean;
  error: string | null;
}

const initialState: AuthState = {
  token: null,
  user: null,
  loading: false,
  error: null,
};

function createAuthStore() {
  const { subscribe, set, update } = writable<AuthState>(initialState);

  return {
    subscribe,
    
    setToken: (token: string) => {
      localStorage.setItem('token', token);
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      
      update(state => ({
        ...state,
        token,
        loading: true,
        error: null,
      }));
      
      // Fetch user data
      axios.get('/api/auth/me')
        .then(res => {
          update(state => ({
            ...state,
            user: res.data,
            loading: false,
          }));
        })
        .catch(err => {
          console.error('Error fetching user data:', err);
          update(state => ({
            ...state,
            loading: false,
            error: 'Failed to fetch user data',
          }));
        });
    },
    
    login: async (email: string, password: string) => {
      update(state => ({ ...state, loading: true, error: null }));
      
      try {
        const res = await axios.post('/api/auth/login', { email, password });
        const { token, ...user } = res.data;
        
        localStorage.setItem('token', token);
        axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
        
        update(state => ({
          ...state,
          token,
          user,
          loading: false,
          error: null,
        }));
        
        return true;
      } catch (err: any) {
        const errorMessage = err.response?.data?.message || 'Login failed';
        
        update(state => ({
          ...state,
          loading: false,
          error: errorMessage,
        }));
        
        return false;
      }
    },
    
    register: async (name: string, email: string, password: string) => {
      update(state => ({ ...state, loading: true, error: null }));
      
      try {
        const res = await axios.post('/api/auth/register', { name, email, password });
        const { token, ...user } = res.data;
        
        localStorage.setItem('token', token);
        axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
        
        update(state => ({
          ...state,
          token,
          user,
          loading: false,
          error: null,
        }));
        
        return true;
      } catch (err: any) {
        const errorMessage = err.response?.data?.message || 'Registration failed';
        
        update(state => ({
          ...state,
          loading: false,
          error: errorMessage,
        }));
        
        return false;
      }
    },
    
    logout: () => {
      localStorage.removeItem('token');
      delete axios.defaults.headers.common['Authorization'];
      set(initialState);
    },
    
    clearError: () => {
      update(state => ({ ...state, error: null }));
    },
  };
}

export const authStore = createAuthStore();

// Set axios default baseURL
axios.defaults.baseURL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Set axios auth header if token exists in localStorage
const token = localStorage.getItem('token');
if (token) {
  axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
}