<script lang="ts">
  import { onMount } from 'svelte';
  
  let reminders = [];
  let loading = true;
  let error = null;
  
  // Form state
  let showForm = false;
  let newReminder = {
    title: '',
    description: '',
    dueDate: '',
    dueTime: '',
    priority: 'medium'
  };
  
  onMount(async () => {
    try {
      // In a real app, this would be an API call
      // const response = await fetch('/api/reminders');
      // reminders = await response.json();
      
      // Placeholder data for development
      reminders = [
        { 
          id: 1, 
          title: 'Team Meeting', 
          description: 'Weekly team sync-up', 
          dueDate: '2025-08-15', 
          dueTime: '10:00', 
          priority: 'high',
          completed: false
        },
        { 
          id: 2, 
          title: 'Project Deadline', 
          description: 'Submit final deliverables', 
          dueDate: '2025-09-01', 
          dueTime: '17:00', 
          priority: 'high',
          completed: false
        },
        { 
          id: 3, 
          title: 'Review Documentation', 
          description: 'Check updated API docs', 
          dueDate: '2025-08-12', 
          dueTime: '14:30', 
          priority: 'medium',
          completed: true
        },
        { 
          id: 4, 
          title: 'Follow up with Client', 
          description: 'Discuss feedback on latest release', 
          dueDate: '2025-08-18', 
          dueTime: '11:00', 
          priority: 'medium',
          completed: false
        }
      ];
      
      loading = false;
    } catch (err) {
      error = err.message;
      loading = false;
    }
  });
  
  function toggleForm() {
    showForm = !showForm;
    if (showForm) {
      // Reset form
      newReminder = {
        title: '',
        description: '',
        dueDate: '',
        dueTime: '',
        priority: 'medium'
      };
    }
  }
  
  function handleSubmit() {
    // In a real app, this would be an API call
    // const response = await fetch('/api/reminders', {
    //   method: 'POST',
    //   headers: { 'Content-Type': 'application/json' },
    //   body: JSON.stringify(newReminder)
    // });
    
    // Simulate adding a new reminder
    const id = Math.max(0, ...reminders.map(r => r.id)) + 1;
    reminders = [
      ...reminders,
      {
        id,
        ...newReminder,
        completed: false
      }
    ];
    
    // Close form
    showForm = false;
  }
  
  function toggleComplete(id: number) {
    reminders = reminders.map(reminder => 
      reminder.id === id 
        ? { ...reminder, completed: !reminder.completed } 
        : reminder
    );
  }
  
  function deleteReminder(id: number) {
    reminders = reminders.filter(reminder => reminder.id !== id);
  }
  
  function getPriorityClass(priority: string): string {
    switch (priority.toLowerCase()) {
      case 'high': return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200';
      case 'medium': return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200';
      case 'low': return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
      default: return 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200';
    }
  }
  
  function formatDateTime(date: string, time: string): string {
    if (!date) return 'No date set';
    
    const dateObj = new Date(`${date}T${time || '00:00'}`);
    const formattedDate = dateObj.toLocaleDateString();
    const formattedTime = time ? dateObj.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '';
    
    return time ? `${formattedDate} at ${formattedTime}` : formattedDate;
  }
  
  function isOverdue(reminder): boolean {
    if (!reminder.dueDate) return false;
    
    const now = new Date();
    const dueDate = new Date(`${reminder.dueDate}T${reminder.dueTime || '23:59'}`);
    
    return !reminder.completed && dueDate < now;
  }
</script>

<div class="container mx-auto">
  <div class="flex justify-between items-center mb-6">
    <h1 class="text-2xl font-bold">Reminders</h1>
    <button 
      on:click={toggleForm}
      class="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md flex items-center"
    >
      {#if showForm}
        Cancel
      {:else}
        <svg class="w-5 h-5 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6"></path>
        </svg>
        Add Reminder
      {/if}
    </button>
  </div>
  
  {#if showForm}
    <div class="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 mb-6">
      <h2 class="text-xl font-semibold mb-4">New Reminder</h2>
      <form on:submit|preventDefault={handleSubmit} class="space-y-4">
        <div>
          <label for="title" class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Title</label>
          <input 
            type="text" 
            id="title" 
            bind:value={newReminder.title} 
            required
            class="w-full px-3 py-2 border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
          />
        </div>
        
        <div>
          <label for="description" class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Description (optional)</label>
          <textarea 
            id="description" 
            bind:value={newReminder.description}
            rows="3" 
            class="w-full px-3 py-2 border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
          ></textarea>
        </div>
        
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label for="dueDate" class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Due Date</label>
            <input 
              type="date" 
              id="dueDate" 
              bind:value={newReminder.dueDate}
              class="w-full px-3 py-2 border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
            />
          </div>
          
          <div>
            <label for="dueTime" class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Due Time (optional)</label>
            <input 
              type="time" 
              id="dueTime" 
              bind:value={newReminder.dueTime}
              class="w-full px-3 py-2 border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
            />
          </div>
        </div>
        
        <div>
          <label for="priority" class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Priority</label>
          <select 
            id="priority" 
            bind:value={newReminder.priority}
            class="w-full px-3 py-2 border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
          >
            <option value="low">Low</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
          </select>
        </div>
        
        <div class="flex justify-end">
          <button 
            type="submit" 
            class="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-md"
          >
            Save Reminder
          </button>
        </div>
      </form>
    </div>
  {/if}
  
  {#if loading}
    <div class="flex justify-center items-center h-64">
      <div class="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
    </div>
  {:else if error}
    <div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
      <p>Error loading reminders: {error}</p>
    </div>
  {:else if reminders.length === 0}
    <div class="bg-gray-100 dark:bg-gray-800 p-6 rounded-lg text-center">
      <p class="text-lg">You don't have any reminders.</p>
      <p class="mt-2">Create a reminder to stay on top of your tasks.</p>
    </div>
  {:else}
    <div class="space-y-4">
      {#each reminders as reminder}
        <div class={`bg-white dark:bg-gray-800 rounded-lg shadow-md p-4 ${reminder.completed ? 'opacity-70' : ''} ${isOverdue(reminder) ? 'border-l-4 border-red-500' : ''}`}>
          <div class="flex items-start justify-between">
            <div class="flex items-start space-x-3">
              <input 
                type="checkbox" 
                checked={reminder.completed}
                on:change={() => toggleComplete(reminder.id)}
                class="mt-1 h-5 w-5 text-blue-600 rounded focus:ring-blue-500"
              />
              <div>
                <h3 class={`font-medium ${reminder.completed ? 'line-through text-gray-500 dark:text-gray-400' : ''}`}>
                  {reminder.title}
                </h3>
                {#if reminder.description}
                  <p class="text-gray-600 dark:text-gray-300 text-sm mt-1">
                    {reminder.description}
                  </p>
                {/if}
                <div class="flex items-center mt-2 space-x-3">
                  <span class="text-sm text-gray-500 dark:text-gray-400">
                    {formatDateTime(reminder.dueDate, reminder.dueTime)}
                  </span>
                  <span class={`text-xs px-2 py-1 rounded-full ${getPriorityClass(reminder.priority)}`}>
                    {reminder.priority.charAt(0).toUpperCase() + reminder.priority.slice(1)}
                  </span>
                  {#if isOverdue(reminder)}
                    <span class="text-xs px-2 py-1 rounded-full bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200">
                      Overdue
                    </span>
                  {/if}
                </div>
              </div>
            </div>
            <button 
              on:click={() => deleteReminder(reminder.id)}
              class="text-gray-400 hover:text-red-500 dark:text-gray-500 dark:hover:text-red-400"
              aria-label="Delete reminder"
            >
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path>
              </svg>
            </button>
          </div>
        </div>
      {/each}
    </div>
  {/if}
</div>