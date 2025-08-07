<script lang="ts">
  import { Router, Route } from 'svelte-navigator';
  import { onMount } from 'svelte';
  import { authStore } from './stores/auth';
  import { themeStore } from './stores/theme';
  
  // Components
  import Navbar from './components/layout/Navbar.svelte';
  import Sidebar from './components/layout/Sidebar.svelte';
  
  // Pages
  import Login from './pages/Login.svelte';
  import Register from './pages/Register.svelte';
  import Dashboard from './pages/Dashboard.svelte';
  import Tasks from './pages/Tasks.svelte';
  import TaskDetail from './pages/TaskDetail.svelte';
  import Goals from './pages/Goals.svelte';
  import GoalDetail from './pages/GoalDetail.svelte';
  import Reminders from './pages/Reminders.svelte';
  import Recommendations from './pages/Recommendations.svelte';
  import Settings from './pages/Settings.svelte';
  import NotFound from './pages/NotFound.svelte';
  
  let isAuthenticated = false;
  let theme: string;
  
  // Subscribe to auth store
  authStore.subscribe(value => {
    isAuthenticated = !!value.token;
  });
  
  // Subscribe to theme store
  themeStore.subscribe(value => {
    theme = value;
    if (typeof document !== 'undefined') {
      if (theme === 'dark') {
        document.documentElement.classList.add('dark');
      } else {
        document.documentElement.classList.remove('dark');
      }
    }
  });
  
  onMount(() => {
    // Check for saved token
    const token = localStorage.getItem('token');
    if (token) {
      authStore.setToken(token);
    }
    
    // Check for saved theme
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
      themeStore.setTheme(savedTheme);
    } else if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
      themeStore.setTheme('dark');
    }
  });
</script>

<Router>
  <div class="min-h-screen bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-100">
    {#if isAuthenticated}
      <Navbar />
      <div class="flex">
        <Sidebar />
        <main class="flex-1 p-6 md:p-8 max-w-7xl mx-auto">
          <Route path="/" component={Dashboard} />
          <Route path="/tasks" component={Tasks} />
          <Route path="/tasks/:id" component={TaskDetail} />
          <Route path="/goals" component={Goals} />
          <Route path="/goals/:id" component={GoalDetail} />
          <Route path="/reminders" component={Reminders} />
          <Route path="/recommendations" component={Recommendations} />
          <Route path="/settings" component={Settings} />
          <Route path="*" component={NotFound} />
        </main>
      </div>
    {:else}
      <div class="min-h-screen flex flex-col">
        <Route path="/" component={Login} />
        <Route path="/register" component={Register} />
        <Route path="*" component={Login} />
      </div>
    {/if}
  </div>
</Router>