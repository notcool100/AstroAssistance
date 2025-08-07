<script lang="ts">
  import { onMount } from 'svelte';
  
  // Placeholder for goals data
  let goals = [];
  let loading = true;
  let error = null;
  
  onMount(async () => {
    try {
      // In a real app, this would be an API call
      // const response = await fetch('/api/goals');
      // goals = await response.json();
      
      // Placeholder data for development
      goals = [
        { id: 1, title: 'Complete Project X', description: 'Finish all tasks related to Project X', progress: 75, dueDate: '2025-12-31' },
        { id: 2, title: 'Learn New Technology', description: 'Master the basics of a new programming language', progress: 30, dueDate: '2025-10-15' },
        { id: 3, title: 'Improve Productivity', description: 'Implement better time management techniques', progress: 50, dueDate: '2025-09-01' }
      ];
      
      loading = false;
    } catch (err) {
      error = err.message;
      loading = false;
    }
  });
  
  function calculateProgressColor(progress: number): string {
    if (progress < 30) return 'bg-red-500';
    if (progress < 70) return 'bg-yellow-500';
    return 'bg-green-500';
  }
</script>

<div class="container mx-auto">
  <div class="flex justify-between items-center mb-6">
    <h1 class="text-2xl font-bold">Goals</h1>
    <button class="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md">
      Add New Goal
    </button>
  </div>
  
  {#if loading}
    <div class="flex justify-center items-center h-64">
      <div class="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
    </div>
  {:else if error}
    <div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
      <p>Error loading goals: {error}</p>
    </div>
  {:else if goals.length === 0}
    <div class="bg-gray-100 dark:bg-gray-800 p-6 rounded-lg text-center">
      <p class="text-lg">You don't have any goals yet.</p>
      <p class="mt-2">Create your first goal to start tracking your progress.</p>
    </div>
  {:else}
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {#each goals as goal}
        <a href={`/goals/${goal.id}`} class="block">
          <div class="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md hover:shadow-lg transition-shadow duration-200">
            <h2 class="text-xl font-semibold mb-2">{goal.title}</h2>
            <p class="text-gray-600 dark:text-gray-300 mb-4">{goal.description}</p>
            
            <div class="mb-2">
              <div class="flex justify-between text-sm mb-1">
                <span>Progress</span>
                <span>{goal.progress}%</span>
              </div>
              <div class="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5">
                <div class={`h-2.5 rounded-full ${calculateProgressColor(goal.progress)}`} style={`width: ${goal.progress}%`}></div>
              </div>
            </div>
            
            <div class="text-sm text-gray-500 dark:text-gray-400 mt-4">
              Due: {new Date(goal.dueDate).toLocaleDateString()}
            </div>
          </div>
        </a>
      {/each}
    </div>
  {/if}
</div>