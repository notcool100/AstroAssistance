<script lang="ts">
  import { Link, useNavigate } from 'svelte-navigator';
  import { authStore } from '../stores/auth';
  
  let email = '';
  let password = '';
  let loading = false;
  let error: string | null = null;
  
  const navigate = useNavigate();
  
  // Subscribe to auth store
  authStore.subscribe(state => {
    loading = state.loading;
    error = state.error;
    
    // Redirect if already authenticated
    if (state.token) {
      navigate('/', { replace: true });
    }
  });
  
  async function handleSubmit() {
    if (!email || !password) {
      error = 'Please enter both email and password';
      return;
    }
    
    const success = await authStore.login(email, password);
    if (success) {
      navigate('/', { replace: true });
    }
  }
</script>

<div class="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900 py-12 px-4 sm:px-6 lg:px-8">
  <div class="max-w-md w-full space-y-8">
    <div>
      <h1 class="text-center text-3xl font-extrabold text-gray-900 dark:text-white">
        AstroAssistance
      </h1>
      <h2 class="mt-6 text-center text-2xl font-bold text-gray-900 dark:text-white">
        Sign in to your account
      </h2>
      <p class="mt-2 text-center text-sm text-gray-600 dark:text-gray-400">
        Or
        <Link to="/register" class="font-medium text-primary-600 hover:text-primary-500 dark:text-primary-400">
          create a new account
        </Link>
      </p>
    </div>
    
    <form class="mt-8 space-y-6" on:submit|preventDefault={handleSubmit}>
      {#if error}
        <div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative dark:bg-red-900 dark:border-red-700 dark:text-red-100" role="alert">
          <span class="block sm:inline">{error}</span>
        </div>
      {/if}
      
      <div class="rounded-md shadow-sm -space-y-px">
        <div>
          <label for="email" class="sr-only">Email address</label>
          <input
            id="email"
            name="email"
            type="email"
            autocomplete="email"
            required
            class="appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 rounded-t-md focus:outline-none focus:ring-primary-500 focus:border-primary-500 focus:z-10 dark:bg-gray-800 dark:border-gray-700 dark:text-white sm:text-sm"
            placeholder="Email address"
            bind:value={email}
          />
        </div>
        <div>
          <label for="password" class="sr-only">Password</label>
          <input
            id="password"
            name="password"
            type="password"
            autocomplete="current-password"
            required
            class="appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 rounded-b-md focus:outline-none focus:ring-primary-500 focus:border-primary-500 focus:z-10 dark:bg-gray-800 dark:border-gray-700 dark:text-white sm:text-sm"
            placeholder="Password"
            bind:value={password}
          />
        </div>
      </div>

      <div>
        <button
          type="submit"
          disabled={loading}
          class="group relative w-full flex justify-center py-2 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {#if loading}
            <span class="absolute left-0 inset-y-0 flex items-center pl-3">
              <svg class="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
            </span>
            Signing in...
          {:else}
            Sign in
          {/if}
        </button>
      </div>
    </form>
  </div>
</div>