<script lang="ts">
  import { onMount } from 'svelte';
  import { useParams, useNavigate, Link } from 'svelte-navigator';
  import { format } from 'date-fns';
  import { tasksStore } from '../stores/tasks';
  
  // Component state
  let task: any = null;
  let loading = false;
  let error: string | null = null;
  let editMode = false;
  let confirmDelete = false;
  
  // Edit form
  let editedTask = {
    title: '',
    description: '',
    category: '',
    priority: '',
    dueDate: '',
    estimatedDuration: 0,
    tags: [] as string[]
  };
  let newTag = '';
  
  const params = useParams();
  const navigate = useNavigate();
  
  // Subscribe to tasks store
  tasksStore.subscribe(state => {
    task = state.currentTask;
    loading = state.loading;
    error = state.error;
    
    if (task && !editMode) {
      // Initialize edited task with current task data
      editedTask = {
        title: task.title,
        description: task.description || '',
        category: task.category,
        priority: task.priority,
        dueDate: task.dueDate ? task.dueDate.split('T')[0] : '',
        estimatedDuration: task.estimatedDuration || 0,
        tags: [...(task.tags || [])]
      };
    }
  });
  
  onMount(() => {
    const taskId = params().id;
    if (taskId) {
      tasksStore.fetchTask(taskId);
    }
  });
  
  function toggleEditMode() {
    editMode = !editMode;
    confirmDelete = false;
  }
  
  function toggleConfirmDelete() {
    confirmDelete = !confirmDelete;
  }
  
  function addTag() {
    if (newTag.trim() && !editedTask.tags.includes(newTag.trim())) {
      editedTask.tags = [...editedTask.tags, newTag.trim()];
      newTag = '';
    }
  }
  
  function removeTag(tag: string) {
    editedTask.tags = editedTask.tags.filter(t => t !== tag);
  }
  
  async function saveTask() {
    if (!task || !editedTask.title) return;
    
    const result = await tasksStore.updateTask(task.id, editedTask);
    
    if (result) {
      editMode = false;
    }
  }
  
  async function deleteTask() {
    if (!task) return;
    
    const success = await tasksStore.deleteTask(task.id);
    
    if (success) {
      navigate('/tasks', { replace: true });
    }
  }
  
  async function toggleCompletion() {
    if (!task) return;
    
    await tasksStore.completeTask(task.id, !task.completed);
  }
  
  function formatDate(dateString: string | null | undefined): string {
    if (!dateString) return 'No due date';
    return format(new Date(dateString), 'MMMM d, yyyy');
  }
  
  function formatDateTime(dateString: string | null | undefined): string {
    if (!dateString) return 'N/A';
    return format(new Date(dateString), 'MMMM d, yyyy h:mm a');
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
  <div class="flex items-center">
    <Link to="/tasks" class="text-primary-600 dark:text-primary-400 hover:underline flex items-center mr-4">
      <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-1" viewBox="0 0 20 20" fill="currentColor">
        <path fill-rule="evenodd" d="M9.707 16.707a1 1 0 01-1.414 0l-6-6a1 1 0 010-1.414l6-6a1 1 0 011.414 1.414L5.414 9H17a1 1 0 110 2H5.414l4.293 4.293a1 1 0 010 1.414z" clip-rule="evenodd" />
      </svg>
      Back to Tasks
    </Link>
    <h1 class="text-2xl font-bold flex-1">Task Details</h1>
  </div>
  
  {#if loading}
    <div class="flex justify-center items-center h-64">
      <div class="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary-500"></div>
    </div>
  {:else if error}
    <div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative dark:bg-red-900 dark:border-red-700 dark:text-red-100" role="alert">
      <span class="block sm:inline">{error}</span>
    </div>
  {:else if !task}
    <div class="card text-center py-12">
      <h2 class="text-xl font-semibold">Task not found</h2>
      <p class="mt-2 text-gray-500 dark:text-gray-400">The task you're looking for doesn't exist or has been deleted.</p>
      <div class="mt-6">
        <Link to="/tasks" class="btn btn-primary">View All Tasks</Link>
      </div>
    </div>
  {:else}
    <div class="card">
      <div class="flex justify-between items-start mb-6">
        <div class="flex items-center">
          <input 
            type="checkbox" 
            class="h-5 w-5 text-primary-600 focus:ring-primary-500 border-gray-300 rounded dark:border-gray-600 dark:bg-gray-700 mr-3"
            checked={task.completed}
            on:change={toggleCompletion}
          />
          {#if !editMode}
            <h2 class={`text-xl font-semibold ${task.completed ? 'line-through text-gray-500 dark:text-gray-400' : ''}`}>
              {task.title}
            </h2>
          {/if}
        </div>
        
        <div class="flex space-x-2">
          {#if !editMode}
            <button 
              class="btn btn-outline flex items-center"
              on:click={toggleEditMode}
            >
              <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 mr-1" viewBox="0 0 20 20" fill="currentColor">
                <path d="M13.586 3.586a2 2 0 112.828 2.828l-.793.793-2.828-2.828.793-.793zM11.379 5.793L3 14.172V17h2.828l8.38-8.379-2.83-2.828z" />
              </svg>
              Edit
            </button>
            <button 
              class="btn btn-outline text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900 flex items-center"
              on:click={toggleConfirmDelete}
            >
              <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 mr-1" viewBox="0 0 20 20" fill="currentColor">
                <path fill-rule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v6a1 1 0 102 0V8a1 1 0 00-1-1z" clip-rule="evenodd" />
              </svg>
              Delete
            </button>
          {/if}
        </div>
      </div>
      
      {#if confirmDelete}
        <div class="bg-red-50 dark:bg-red-900 border border-red-200 dark:border-red-700 rounded-md p-4 mb-6">
          <div class="flex">
            <div class="flex-shrink-0">
              <svg class="h-5 w-5 text-red-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd" />
              </svg>
            </div>
            <div class="ml-3">
              <h3 class="text-sm font-medium text-red-800 dark:text-red-200">Are you sure you want to delete this task?</h3>
              <div class="mt-2 text-sm text-red-700 dark:text-red-300">
                <p>This action cannot be undone.</p>
              </div>
              <div class="mt-4 flex space-x-3">
                <button 
                  type="button" 
                  class="btn btn-outline"
                  on:click={toggleConfirmDelete}
                >
                  Cancel
                </button>
                <button 
                  type="button" 
                  class="btn bg-red-600 hover:bg-red-700 text-white"
                  on:click={deleteTask}
                >
                  Delete
                </button>
              </div>
            </div>
          </div>
        </div>
      {/if}
      
      {#if editMode}
        <form on:submit|preventDefault={saveTask} class="space-y-4">
          <div>
            <label for="title" class="block text-sm font-medium text-gray-700 dark:text-gray-300">Title</label>
            <input 
              type="text" 
              id="title" 
              class="input w-full mt-1" 
              bind:value={editedTask.title} 
              required
            />
          </div>
          
          <div>
            <label for="description" class="block text-sm font-medium text-gray-700 dark:text-gray-300">Description</label>
            <textarea 
              id="description" 
              class="input w-full mt-1" 
              rows="3" 
              bind:value={editedTask.description}
            ></textarea>
          </div>
          
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label for="category" class="block text-sm font-medium text-gray-700 dark:text-gray-300">Category</label>
              <select id="category" class="input w-full mt-1" bind:value={editedTask.category}>
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
              <select id="priority" class="input w-full mt-1" bind:value={editedTask.priority}>
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
                bind:value={editedTask.dueDate}
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
                bind:value={editedTask.estimatedDuration}
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
                bind:value={newTag}
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
            
            {#if editedTask.tags.length > 0}
              <div class="flex flex-wrap gap-2 mt-2">
                {#each editedTask.tags as tag}
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
          
          <div class="flex justify-end space-x-3 pt-4">
            <button 
              type="button" 
              class="btn btn-outline"
              on:click={toggleEditMode}
            >
              Cancel
            </button>
            <button 
              type="submit" 
              class="btn btn-primary"
              disabled={!editedTask.title}
            >
              Save Changes
            </button>
          </div>
        </form>
      {:else}
        <div class="space-y-6">
          <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 class="text-sm font-medium text-gray-500 dark:text-gray-400">Category</h3>
              <p class="mt-1 text-sm text-gray-900 dark:text-white">
                {task.category.charAt(0).toUpperCase() + task.category.slice(1)}
              </p>
            </div>
            
            <div>
              <h3 class="text-sm font-medium text-gray-500 dark:text-gray-400">Priority</h3>
              <p class="mt-1">
                <span class={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${getPriorityClass(task.priority)}`}>
                  {task.priority.charAt(0).toUpperCase() + task.priority.slice(1)}
                </span>
              </p>
            </div>
            
            <div>
              <h3 class="text-sm font-medium text-gray-500 dark:text-gray-400">Due Date</h3>
              <p class="mt-1 text-sm text-gray-900 dark:text-white">
                {formatDate(task.dueDate)}
              </p>
            </div>
            
            <div>
              <h3 class="text-sm font-medium text-gray-500 dark:text-gray-400">Estimated Duration</h3>
              <p class="mt-1 text-sm text-gray-900 dark:text-white">
                {task.estimatedDuration ? `${task.estimatedDuration} minutes` : 'Not specified'}
              </p>
            </div>
            
            <div>
              <h3 class="text-sm font-medium text-gray-500 dark:text-gray-400">Created</h3>
              <p class="mt-1 text-sm text-gray-900 dark:text-white">
                {formatDateTime(task.createdAt)}
              </p>
            </div>
            
            <div>
              <h3 class="text-sm font-medium text-gray-500 dark:text-gray-400">Last Updated</h3>
              <p class="mt-1 text-sm text-gray-900 dark:text-white">
                {formatDateTime(task.updatedAt)}
              </p>
            </div>
            
            {#if task.completed}
              <div>
                <h3 class="text-sm font-medium text-gray-500 dark:text-gray-400">Completed</h3>
                <p class="mt-1 text-sm text-gray-900 dark:text-white">
                  {formatDateTime(task.completedAt)}
                </p>
              </div>
            {/if}
          </div>
          
          <div>
            <h3 class="text-sm font-medium text-gray-500 dark:text-gray-400">Description</h3>
            <div class="mt-1 text-sm text-gray-900 dark:text-white prose dark:prose-invert max-w-none">
              {#if task.description}
                <p>{task.description}</p>
              {:else}
                <p class="text-gray-500 dark:text-gray-400">No description provided</p>
              {/if}
            </div>
          </div>
          
          <div>
            <h3 class="text-sm font-medium text-gray-500 dark:text-gray-400">Tags</h3>
            <div class="mt-1">
              {#if task.tags && task.tags.length > 0}
                <div class="flex flex-wrap gap-2">
                  {#each task.tags as tag}
                    <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200">
                      {tag}
                    </span>
                  {/each}
                </div>
              {:else}
                <p class="text-sm text-gray-500 dark:text-gray-400">No tags</p>
              {/if}
            </div>
          </div>
        </div>
      {/if}
    </div>
  {/if}
</div>