<script lang="ts">
  import { onMount } from 'svelte';
  import { useParams } from 'svelte-navigator';
  
  const params = useParams();
  
  let goal = null;
  let relatedTasks = [];
  let loading = true;
  let error = null;
  
  onMount(async () => {
    try {
      const goalId = $params.id;
      
      // In a real app, these would be API calls
      // const goalResponse = await fetch(`/api/goals/${goalId}`);
      // goal = await goalResponse.json();
      // const tasksResponse = await fetch(`/api/goals/${goalId}/tasks`);
      // relatedTasks = await tasksResponse.json();
      
      // Placeholder data for development
      goal = {
        id: parseInt(goalId),
        title: 'Complete Project X',
        description: 'Finish all tasks related to Project X by the end of the year. This includes planning, development, testing, and deployment phases.',
        progress: 75,
        dueDate: '2025-12-31',
        createdAt: '2025-01-15',
        category: 'Work',
        priority: 'High'
      };
      
      relatedTasks = [
        { id: 101, title: 'Create project plan', completed: true, dueDate: '2025-02-15' },
        { id: 102, title: 'Develop core features', completed: true, dueDate: '2025-06-30' },
        { id: 103, title: 'Conduct user testing', completed: false, dueDate: '2025-09-15' },
        { id: 104, title: 'Fix reported issues', completed: false, dueDate: '2025-11-01' },
        { id: 105, title: 'Deploy to production', completed: false, dueDate: '2025-12-15' }
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
  
  function formatDate(dateString: string): string {
    return new Date(dateString).toLocaleDateString();
  }
</script>

<div class="container mx-auto">
  {#if loading}
    <div class="flex justify-center items-center h-64">
      <div class="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
    </div>
  {:else if error}
    <div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
      <p>Error loading goal: {error}</p>
    </div>
  {:else if goal}
    <div class="mb-6">
      <a href="/goals" class="text-blue-600 dark:text-blue-400 hover:underline inline-flex items-center">
        <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path>
        </svg>
        Back to Goals
      </a>
    </div>
    
    <div class="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 mb-8">
      <div class="flex justify-between items-start mb-4">
        <h1 class="text-2xl font-bold">{goal.title}</h1>
        <div class="flex space-x-2">
          <button class="bg-blue-600 hover:bg-blue-700 text-white px-3 py-1 rounded-md">
            Edit
          </button>
          <button class="bg-red-600 hover:bg-red-700 text-white px-3 py-1 rounded-md">
            Delete
          </button>
        </div>
      </div>
      
      <p class="text-gray-600 dark:text-gray-300 mb-6">{goal.description}</p>
      
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        <div>
          <h2 class="text-lg font-semibold mb-3">Details</h2>
          <div class="space-y-2">
            <div class="flex justify-between">
              <span class="text-gray-500 dark:text-gray-400">Category:</span>
              <span>{goal.category}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-500 dark:text-gray-400">Priority:</span>
              <span>{goal.priority}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-500 dark:text-gray-400">Due Date:</span>
              <span>{formatDate(goal.dueDate)}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-500 dark:text-gray-400">Created:</span>
              <span>{formatDate(goal.createdAt)}</span>
            </div>
          </div>
        </div>
        
        <div>
          <h2 class="text-lg font-semibold mb-3">Progress</h2>
          <div class="mb-2">
            <div class="flex justify-between text-sm mb-1">
              <span>Overall Completion</span>
              <span>{goal.progress}%</span>
            </div>
            <div class="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5">
              <div class={`h-2.5 rounded-full ${calculateProgressColor(goal.progress)}`} style={`width: ${goal.progress}%`}></div>
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <div class="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
      <div class="flex justify-between items-center mb-4">
        <h2 class="text-xl font-semibold">Related Tasks</h2>
        <button class="bg-green-600 hover:bg-green-700 text-white px-3 py-1 rounded-md">
          Add Task
        </button>
      </div>
      
      {#if relatedTasks.length === 0}
        <p class="text-gray-500 dark:text-gray-400">No tasks associated with this goal.</p>
      {:else}
        <div class="space-y-3">
          {#each relatedTasks as task}
            <div class="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-md">
              <div class="flex items-center">
                <input type="checkbox" checked={task.completed} class="mr-3 h-4 w-4 text-blue-600 rounded focus:ring-blue-500" />
                <span class={task.completed ? 'line-through text-gray-400' : ''}>{task.title}</span>
              </div>
              <div class="flex items-center">
                <span class="text-sm text-gray-500 dark:text-gray-400 mr-4">Due: {formatDate(task.dueDate)}</span>
                <a href={`/tasks/${task.id}`} class="text-blue-600 dark:text-blue-400 hover:underline">View</a>
              </div>
            </div>
          {/each}
        </div>
      {/if}
    </div>
  {:else}
    <div class="bg-yellow-100 border border-yellow-400 text-yellow-700 px-4 py-3 rounded">
      <p>Goal not found.</p>
    </div>
  {/if}
</div>