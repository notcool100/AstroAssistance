<script lang="ts">
  import { onMount } from 'svelte';
  import { Link, useNavigate } from 'svelte-navigator';
  import { format } from 'date-fns';
  import { tasksStore } from '../stores/tasks';
  
  // Component state
  let tasks: any[] = [];
  let loading = false;
  let error: string | null = null;
  let pagination = {
    page: 1,
    limit: 10,
    total: 0,
    pages: 0
  };
  
  // Filters
  let filters = {
    search: '',
    category: '',
    priority: '',
    completed: '',
    page: 1
  };
  
  // New task form
  let showNewTaskForm = false;
  let newTask = {
    title: '',
    description: '',
    category: 'work',
    priority: 'medium',
    dueDate: '',
    estimatedDuration: 30,
    tags: [] as string[]
  };
  let newTaskTag = '';
  
  const navigate = useNavigate();
  
  // Subscribe to tasks store
  tasksStore.subscribe(state => {
    tasks = state.tasks;
    loading = state.loading;
    error = state.error;
    pagination = state.pagination;
  });
  
  onMount(() => {
    loadTasks();
  });
  
  function loadTasks() {
    tasksStore.fetchTasks(filters);
  }
  
  function handleFilterChange() {
    filters.page = 1;
    loadTasks();
  }
  
  function handlePageChange(newPage: number) {
    if (newPage < 1 || newPage > pagination.pages) return;
    filters.page = newPage;
    loadTasks();
  }
  
  function toggleNewTaskForm() {
    showNewTaskForm = !showNewTaskForm;
  }
  
  function addTag() {
    if (newTaskTag.trim() && !newTask.tags.includes(newTaskTag.trim())) {
      newTask.tags = [...newTask.tags, newTaskTag.trim()];
      newTaskTag = '';
    }
  }
  
  function removeTag(tag: string) {
    newTask.tags = newTask.tags.filter(t => t !== tag);
  }
  
  async function createTask() {
    if (!newTask.title) return;
    
    const result = await tasksStore.createTask(newTask);
    
    if (result) {
      // Reset form
      newTask = {
        title: '',
        description: '',
        category: 'work',
        priority: 'medium',
        dueDate: '',
        estimatedDuration: 30,
        tags: []
      };
      showNewTaskForm = false;
      loadTasks();
    }
  }
  
  async function toggleTaskCompletion(id: string, completed: boolean) {
    await tasksStore.completeTask(id, !completed);
  }
  
  function formatDate(dateString: string | null | undefined): string {
    if (!dateString) return 'No due date';
    return format(new Date(dateString), 'MMM d, yyyy');
  }
  
  function getPriorityClass(priority: string): string {
    switch (priority) {
      case 'high':
        return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200';
      case 'medium':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200';
      default:
        return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
    }
  }
</script>

<div class="space-y-6">
  <div class="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
    <h1 class="text-2xl font-bold">Tasks</h1>
    <button 
      class="btn btn-primary flex items-center"
      on:click={toggleNewTaskForm}
    >
      <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
        <path fill-rule="evenodd" d="M10 5a1 1 0 011 1v3h3a1 1 0 110 2h-3v3a1 1 0 11-2 0v-3H6a1 1 0 110-2h3V6a1 1 0 011-1z" clip-rule="evenodd" />
      </svg>
      New Task
    </button>
  </div>
  
  {#if error}
    <div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative dark:bg-red-900 dark:border-red-700 dark:text-red-100" role="alert">
      <span class="block sm:inline">{error}</span>
    </div>
  {/if}
  
  {#if showNewTaskForm}
    <div class="card">
      <h2 class="text-xl font-semibold mb-4">Create New Task</h2>
      <form on:submit|preventDefault={createTask} class="space-y-4">
        <div>
          <label for="title" class="block text-sm font-medium text-gray-700 dark:text-gray-300">Title</label>
          <input 
            type="text" 
            id="title" 
            class="input w-full mt-1" 
            bind:value={newTask.title} 
            required
          />
        </div>
        
        <div>
          <label for="description" class="block text-sm font-medium text-gray-700 dark:text-gray-300">Description</label>
          <textarea 
            id="description" 
            class="input w-full mt-1" 
            rows="3" 
            bind:value={newTask.description}
          ></textarea>
        </div>
        
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label for="category" class="block text-sm font-medium text-gray-700 dark:text-gray-300">Category</label>
            <select id="category" class="input w-full mt-1" bind:value={newTask.category}>
              <option value="work">Work</option>
              <option value="personal">Personal</option>
              <option value="health">Health</option>
              <option value="education">Education</option>
              <option value="finance">Finance</option>
              <option value="other">Other</option>
            </select>
          </div>
          
          <div>
            <label for="priority" class="block text-sm font-medium text-gray-700 dark:text-gray-300">Priority</label>
            <select id="priority" class="input w-full mt-1" bind:value={newTask.priority}>
              <option value="high">High</option>
              <option value="medium">Medium</option>
              <option value="low">Low</option>
            </select>
          </div>
          
          <div>
            <label for="dueDate" class="block text-sm font-medium text-gray-700 dark:text-gray-300">Due Date</label>
            <input 
              type="date" 
              id="dueDate" 
              class="input w-full mt-1" 
              bind:value={newTask.dueDate}
            />
          </div>
          
          <div>
            <label for="estimatedDuration" class="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Estimated Duration (minutes)
            </label>
            <input 
              type="number" 
              id="estimatedDuration" 
              class="input w-full mt-1" 
              min="1" 
              bind:value={newTask.estimatedDuration}
            />
          </div>
        </div>
        
        <div>
          <label for="tags" class="block text-sm font-medium text-gray-700 dark:text-gray-300">Tags</label>
          <div class="flex mt-1">
            <input 
              type="text" 
              id="tags" 
              class="input flex-1 rounded-r-none" 
              bind:value={newTaskTag}
              placeholder="Add a tag"
            />
            <button 
              type="button" 
              class="px-4 py-2 bg-gray-200 dark:bg-gray-700 rounded-r-md hover:bg-gray-300 dark:hover:bg-gray-600"
              on:click={addTag}
            >
              Add
            </button>
          </div>
          
          {#if newTask.tags.length > 0}
            <div class="flex flex-wrap gap-2 mt-2">
              {#each newTask.tags as tag}
                <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200">
                  {tag}
                  <button 
                    type="button" 
                    class="ml-1.5 inline-flex items-center justify-center h-4 w-4 rounded-full text-blue-400 hover:text-blue-500 dark:text-blue-300 dark:hover:text-blue-200"
                    on:click={() => removeTag(tag)}
                  >
                    <svg class="h-3 w-3" fill="currentColor" viewBox="0 0 20 20">
                      <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd" />
                    </svg>
                  </button>
                </span>
              {/each}
            </div>
          {/if}
        </div>
        
        <div class="flex justify-end space-x-3">
          <button 
            type="button" 
            class="btn btn-outline"
            on:click={toggleNewTaskForm}
          >
            Cancel
          </button>
          <button 
            type="submit" 
            class="btn btn-primary"
            disabled={!newTask.title}
          >
            Create Task
          </button>
        </div>
      </form>
    </div>
  {/if}
  
  <!-- Filters -->
  <div class="card">
    <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
      <div>
        <label for="search" class="block text-sm font-medium text-gray-700 dark:text-gray-300">Search</label>
        <input 
          type="text" 
          id="search" 
          class="input w-full mt-1" 
          placeholder="Search tasks..." 
          bind:value={filters.search} 
          on:input={handleFilterChange}
        />
      </div>
      
      <div>
        <label for="category-filter" class="block text-sm font-medium text-gray-700 dark:text-gray-300">Category</label>
        <select 
          id="category-filter" 
          class="input w-full mt-1" 
          bind:value={filters.category} 
          on:change={handleFilterChange}
        >
          <option value="">All Categories</option>
          <option value="work">Work</option>
          <option value="personal">Personal</option>
          <option value="health">Health</option>
          <option value="education">Education</option>
          <option value="finance">Finance</option>
          <option value="other">Other</option>
        </select>
      </div>
      
      <div>
        <label for="priority-filter" class="block text-sm font-medium text-gray-700 dark:text-gray-300">Priority</label>
        <select 
          id="priority-filter" 
          class="input w-full mt-1" 
          bind:value={filters.priority} 
          on:change={handleFilterChange}
        >
          <option value="">All Priorities</option>
          <option value="high">High</option>
          <option value="medium">Medium</option>
          <option value="low">Low</option>
        </select>
      </div>
      
      <div>
        <label for="completed-filter" class="block text-sm font-medium text-gray-700 dark:text-gray-300">Status</label>
        <select 
          id="completed-filter" 
          class="input w-full mt-1" 
          bind:value={filters.completed} 
          on:change={handleFilterChange}
        >
          <option value="">All Tasks</option>
          <option value="false">Pending</option>
          <option value="true">Completed</option>
        </select>
      </div>
    </div>
  </div>
  
  <!-- Tasks List -->
  <div class="card">
    {#if loading}
      <div class="flex justify-center items-center h-64">
        <div class="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary-500"></div>
      </div>
    {:else if tasks.length === 0}
      <div class="text-center py-12">
        <svg xmlns="http://www.w3.org/2000/svg" class="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
        </svg>
        <h3 class="mt-2 text-sm font-medium text-gray-900 dark:text-white">No tasks found</h3>
        <p class="mt-1 text-sm text-gray-500 dark:text-gray-400">
          {filters.search || filters.category || filters.priority || filters.completed 
            ? 'Try changing your filters or create a new task.' 
            : 'Get started by creating a new task.'}
        </p>
        <div class="mt-6">
          <button 
            type="button" 
            class="btn btn-primary"
            on:click={toggleNewTaskForm}
          >
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
              <path fill-rule="evenodd" d="M10 5a1 1 0 011 1v3h3a1 1 0 110 2h-3v3a1 1 0 11-2 0v-3H6a1 1 0 110-2h3V6a1 1 0 011-1z" clip-rule="evenodd" />
            </svg>
            New Task
          </button>
        </div>
      </div>
    {:else}
      <div class="overflow-x-auto">
        <table class="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
          <thead class="bg-gray-50 dark:bg-gray-800">
            <tr>
              <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Status
              </th>
              <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Task
              </th>
              <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Category
              </th>
              <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Priority
              </th>
              <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Due Date
              </th>
              <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Duration
              </th>
            </tr>
          </thead>
          <tbody class="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
            {#each tasks as task}
              <tr class={task.completed ? 'bg-gray-50 dark:bg-gray-900' : ''}>
                <td class="px-6 py-4 whitespace-nowrap">
                  <input 
                    type="checkbox" 
                    class="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded dark:border-gray-600 dark:bg-gray-700"
                    checked={task.completed}
                    on:change={() => toggleTaskCompletion(task.id, task.completed)}
                  />
                </td>
                <td class="px-6 py-4">
                  <div class="flex items-center">
                    <div>
                      <div class={task.completed ? 'line-through text-gray-500 dark:text-gray-400' : 'font-medium text-gray-900 dark:text-white'}>
                        <Link to={`/tasks/${task.id}`} class="hover:text-primary-600 dark:hover:text-primary-400">
                          {task.title}
                        </Link>
                      </div>
                      {#if task.description}
                        <div class="text-sm text-gray-500 dark:text-gray-400 truncate max-w-xs">
                          {task.description}
                        </div>
                      {/if}
                      {#if task.tags && task.tags.length > 0}
                        <div class="flex flex-wrap gap-1 mt-1">
                          {#each task.tags as tag}
                            <span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200">
                              {tag}
                            </span>
                          {/each}
                        </div>
                      {/if}
                    </div>
                  </div>
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                  {task.category.charAt(0).toUpperCase() + task.category.slice(1)}
                </td>
                <td class="px-6 py-4 whitespace-nowrap">
                  <span class={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${getPriorityClass(task.priority)}`}>
                    {task.priority.charAt(0).toUpperCase() + task.priority.slice(1)}
                  </span>
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                  {formatDate(task.dueDate)}
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                  {task.estimatedDuration ? `${task.estimatedDuration} min` : 'N/A'}
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
      
      <!-- Pagination -->
      {#if pagination.pages > 1}
        <div class="flex items-center justify-between border-t border-gray-200 dark:border-gray-700 px-4 py-3 sm:px-6 mt-4">
          <div class="flex-1 flex justify-between sm:hidden">
            <button 
              class="btn btn-outline"
              disabled={pagination.page === 1}
              on:click={() => handlePageChange(pagination.page - 1)}
            >
              Previous
            </button>
            <button 
              class="btn btn-outline ml-3"
              disabled={pagination.page === pagination.pages}
              on:click={() => handlePageChange(pagination.page + 1)}
            >
              Next
            </button>
          </div>
          <div class="hidden sm:flex-1 sm:flex sm:items-center sm:justify-between">
            <div>
              <p class="text-sm text-gray-700 dark:text-gray-300">
                Showing <span class="font-medium">{(pagination.page - 1) * pagination.limit + 1}</span> to <span class="font-medium">{Math.min(pagination.page * pagination.limit, pagination.total)}</span> of <span class="font-medium">{pagination.total}</span> results
              </p>
            </div>
            <div>
              <nav class="relative z-0 inline-flex rounded-md shadow-sm -space-x-px" aria-label="Pagination">
                <button 
                  class="relative inline-flex items-center px-2 py-2 rounded-l-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-sm font-medium text-gray-500 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-gray-700"
                  disabled={pagination.page === 1}
                  on:click={() => handlePageChange(pagination.page - 1)}
                >
                  <span class="sr-only">Previous</span>
                  <svg class="h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                    <path fill-rule="evenodd" d="M12.707 5.293a1 1 0 010 1.414L9.414 10l3.293 3.293a1 1 0 01-1.414 1.414l-4-4a1 1 0 010-1.414l4-4a1 1 0 011.414 0z" clip-rule="evenodd" />
                  </svg>
                </button>
                
                {#each Array(pagination.pages) as _, i}
                  {#if i + 1 === pagination.page || i + 1 === 1 || i + 1 === pagination.pages || (i + 1 >= pagination.page - 1 && i + 1 <= pagination.page + 1)}
                    <button 
                      class={`relative inline-flex items-center px-4 py-2 border text-sm font-medium ${
                        i + 1 === pagination.page 
                          ? 'z-10 bg-primary-50 dark:bg-primary-900 border-primary-500 dark:border-primary-500 text-primary-600 dark:text-primary-200' 
                          : 'bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600 text-gray-500 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-gray-700'
                      }`}
                      on:click={() => handlePageChange(i + 1)}
                    >
                      {i + 1}
                    </button>
                  {:else if (i + 1 === 2 && pagination.page > 3) || (i + 1 === pagination.pages - 1 && pagination.page < pagination.pages - 2)}
                    <span class="relative inline-flex items-center px-4 py-2 border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-sm font-medium text-gray-700 dark:text-gray-300">
                      ...
                    </span>
                  {/if}
                {/each}
                
                <button 
                  class="relative inline-flex items-center px-2 py-2 rounded-r-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-sm font-medium text-gray-500 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-gray-700"
                  disabled={pagination.page === pagination.pages}
                  on:click={() => handlePageChange(pagination.page + 1)}
                >
                  <span class="sr-only">Next</span>
                  <svg class="h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                    <path fill-rule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clip-rule="evenodd" />
                  </svg>
                </button>
              </nav>
            </div>
          </div>
        </div>
      {/if}
    {/if}
  </div>
</div>