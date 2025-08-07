<script lang="ts">
  import { onMount } from 'svelte';
  import { Link } from 'svelte-navigator';
  import axios from 'axios';
  import { format } from 'date-fns';
  import { Doughnut } from 'svelte-chartjs';
  import { Chart, Title, Tooltip, Legend, ArcElement } from 'chart.js';
  
  // Register Chart.js components
  Chart.register(Title, Tooltip, Legend, ArcElement);
  
  // Dashboard data
  let loading = true;
  let error: string | null = null;
  let tasks: any[] = [];
  let upcomingTasks: any[] = [];
  let goals: any[] = [];
  let recommendations: any[] = [];
  let taskStats = {
    completed: 0,
    pending: 0,
    overdue: 0
  };
  
  // Chart data
  let taskChartData = {
    labels: ['Completed', 'Pending', 'Overdue'],
    datasets: [
      {
        data: [0, 0, 0],
        backgroundColor: ['#10B981', '#3B82F6', '#EF4444'],
        hoverBackgroundColor: ['#059669', '#2563EB', '#DC2626'],
      },
    ],
  };
  
  // Chart options
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'bottom' as const,
      },
      title: {
        display: true,
        text: 'Task Status',
      },
    },
  };
  
  onMount(async () => {
    try {
      loading = true;
      
      // Fetch tasks, goals, and recommendations in parallel
      const [tasksRes, goalsRes, recommendationsRes] = await Promise.all([
        axios.get('/api/tasks?limit=10'),
        axios.get('/api/goals?limit=5'),
        axios.get('/api/recommendations?limit=3')
      ]);
      
      tasks = tasksRes.data.tasks;
      goals = goalsRes.data.goals;
      recommendations = recommendationsRes.data.recommendations;
      
      // Calculate task statistics
      const now = new Date();
      taskStats.completed = tasks.filter(task => task.completed).length;
      taskStats.overdue = tasks.filter(task => !task.completed && task.dueDate && new Date(task.dueDate) < now).length;
      taskStats.pending = tasks.filter(task => !task.completed && (!task.dueDate || new Date(task.dueDate) >= now)).length;
      
      // Update chart data
      taskChartData.datasets[0].data = [taskStats.completed, taskStats.pending, taskStats.overdue];
      
      // Get upcoming tasks (due in the next 3 days)
      const threeDaysFromNow = new Date();
      threeDaysFromNow.setDate(threeDaysFromNow.getDate() + 3);
      
      upcomingTasks = tasks.filter(task => 
        !task.completed && 
        task.dueDate && 
        new Date(task.dueDate) >= now && 
        new Date(task.dueDate) <= threeDaysFromNow
      ).sort((a, b) => new Date(a.dueDate).getTime() - new Date(b.dueDate).getTime());
      
      loading = false;
    } catch (err: any) {
      console.error('Error fetching dashboard data:', err);
      error = 'Failed to load dashboard data';
      loading = false;
    }
  });
  
  function formatDate(dateString: string): string {
    return format(new Date(dateString), 'MMM d, yyyy');
  }
  
  async function applyRecommendation(id: string) {
    try {
      await axios.put(`/api/recommendations/${id}/apply`, { applied: true });
      recommendations = recommendations.map(rec => 
        rec.id === id ? { ...rec, applied: true, appliedAt: new Date().toISOString() } : rec
      );
    } catch (err) {
      console.error('Error applying recommendation:', err);
    }
  }
</script>

<div class="space-y-6">
  <div class="flex justify-between items-center">
    <h1 class="text-2xl font-bold">Dashboard</h1>
    <div class="text-sm text-gray-500 dark:text-gray-400">
      {format(new Date(), 'EEEE, MMMM d, yyyy')}
    </div>
  </div>
  
  {#if loading}
    <div class="flex justify-center items-center h-64">
      <div class="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary-500"></div>
    </div>
  {:else if error}
    <div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative dark:bg-red-900 dark:border-red-700 dark:text-red-100" role="alert">
      <span class="block sm:inline">{error}</span>
    </div>
  {:else}
    <!-- Stats Cards -->
    <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
      <div class="card bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900 dark:to-blue-800 border border-blue-200 dark:border-blue-700">
        <div class="flex justify-between items-start">
          <div>
            <h3 class="text-lg font-semibold text-blue-700 dark:text-blue-300">Total Tasks</h3>
            <p class="text-3xl font-bold mt-2">{tasks.length}</p>
          </div>
          <div class="p-2 bg-blue-200 dark:bg-blue-700 rounded-lg">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-blue-700 dark:text-blue-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
            </svg>
          </div>
        </div>
        <div class="mt-4 flex justify-between text-sm">
          <span class="text-blue-600 dark:text-blue-300">{taskStats.completed} Completed</span>
          <span class="text-yellow-600 dark:text-yellow-300">{taskStats.pending} Pending</span>
          <span class="text-red-600 dark:text-red-300">{taskStats.overdue} Overdue</span>
        </div>
      </div>
      
      <div class="card bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900 dark:to-purple-800 border border-purple-200 dark:border-purple-700">
        <div class="flex justify-between items-start">
          <div>
            <h3 class="text-lg font-semibold text-purple-700 dark:text-purple-300">Active Goals</h3>
            <p class="text-3xl font-bold mt-2">{goals.filter(goal => !goal.completed).length}</p>
          </div>
          <div class="p-2 bg-purple-200 dark:bg-purple-700 rounded-lg">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-purple-700 dark:text-purple-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </div>
        </div>
        <div class="mt-4">
          <Link to="/goals" class="text-sm text-purple-600 dark:text-purple-300 hover:underline">View all goals →</Link>
        </div>
      </div>
      
      <div class="card bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900 dark:to-green-800 border border-green-200 dark:border-green-700">
        <div class="flex justify-between items-start">
          <div>
            <h3 class="text-lg font-semibold text-green-700 dark:text-green-300">Recommendations</h3>
            <p class="text-3xl font-bold mt-2">{recommendations.length}</p>
          </div>
          <div class="p-2 bg-green-200 dark:bg-green-700 rounded-lg">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-green-700 dark:text-green-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
          </div>
        </div>
        <div class="mt-4">
          <Link to="/recommendations" class="text-sm text-green-600 dark:text-green-300 hover:underline">View all recommendations →</Link>
        </div>
      </div>
    </div>
    
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <!-- Task Chart -->
      <div class="card lg:col-span-1">
        <h2 class="text-xl font-semibold mb-4">Task Overview</h2>
        <div class="h-64">
          <Doughnut data={taskChartData} options={chartOptions} />
        </div>
      </div>
      
      <!-- Upcoming Tasks -->
      <div class="card lg:col-span-2">
        <div class="flex justify-between items-center mb-4">
          <h2 class="text-xl font-semibold">Upcoming Tasks</h2>
          <Link to="/tasks" class="text-sm text-primary-600 dark:text-primary-400 hover:underline">View all</Link>
        </div>
        
        {#if upcomingTasks.length === 0}
          <p class="text-gray-500 dark:text-gray-400 py-4">No upcoming tasks in the next 3 days.</p>
        {:else}
          <div class="overflow-x-auto">
            <table class="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead class="bg-gray-50 dark:bg-gray-800">
                <tr>
                  <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Task</th>
                  <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Due Date</th>
                  <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Priority</th>
                </tr>
              </thead>
              <tbody class="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                {#each upcomingTasks as task}
                  <tr>
                    <td class="px-6 py-4 whitespace-nowrap">
                      <Link to={`/tasks/${task.id}`} class="text-primary-600 dark:text-primary-400 hover:underline">{task.title}</Link>
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                      {task.dueDate ? formatDate(task.dueDate) : 'No due date'}
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap">
                      <span class={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full 
                        ${task.priority === 'high' ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200' : 
                          task.priority === 'medium' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200' : 
                          'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'}`}>
                        {task.priority.charAt(0).toUpperCase() + task.priority.slice(1)}
                      </span>
                    </td>
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
        {/if}
      </div>
    </div>
    
    <!-- Recommendations -->
    <div class="card">
      <div class="flex justify-between items-center mb-4">
        <h2 class="text-xl font-semibold">Personalized Recommendations</h2>
        <Link to="/recommendations" class="text-sm text-primary-600 dark:text-primary-400 hover:underline">View all</Link>
      </div>
      
      {#if recommendations.length === 0}
        <p class="text-gray-500 dark:text-gray-400 py-4">No recommendations available.</p>
      {:else}
        <div class="space-y-4">
          {#each recommendations as recommendation}
            <div class="p-4 border rounded-lg {recommendation.applied ? 'border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800' : 'border-primary-200 dark:border-primary-700 bg-primary-50 dark:bg-primary-900'}">
              <div class="flex justify-between">
                <div class="flex items-start space-x-3">
                  <div class={`p-2 rounded-full ${recommendation.type === 'task' ? 'bg-blue-100 dark:bg-blue-800' : 
                    recommendation.type === 'break' ? 'bg-green-100 dark:bg-green-800' : 
                    'bg-purple-100 dark:bg-purple-800'}`}>
                    {#if recommendation.type === 'task'}
                      <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-blue-600 dark:text-blue-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
                      </svg>
                    {:else if recommendation.type === 'break'}
                      <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-green-600 dark:text-green-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    {:else}
                      <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-purple-600 dark:text-purple-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
                      </svg>
                    {/if}
                  </div>
                  <div>
                    <p class="font-medium">{recommendation.content}</p>
                    {#if recommendation.reason}
                      <p class="text-sm text-gray-500 dark:text-gray-400 mt-1">{recommendation.reason}</p>
                    {/if}
                  </div>
                </div>
                {#if !recommendation.applied}
                  <button 
                    class="px-3 py-1 text-sm bg-primary-600 text-white rounded hover:bg-primary-700 transition-colors"
                    on:click={() => applyRecommendation(recommendation.id)}
                  >
                    Apply
                  </button>
                {:else}
                  <span class="text-sm text-gray-500 dark:text-gray-400">Applied</span>
                {/if}
              </div>
            </div>
          {/each}
        </div>
      {/if}
    </div>
  {/if}
</div>