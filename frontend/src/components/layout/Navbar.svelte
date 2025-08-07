<script lang="ts">
  import { Link, useNavigate } from 'svelte-navigator';
  import { authStore } from '../../stores/auth';
  import { themeStore } from '../../stores/theme';
  
  let user: any = null;
  let theme: string;
  let isMenuOpen = false;
  
  const navigate = useNavigate();
  
  // Subscribe to auth store
  authStore.subscribe(state => {
    user = state.user;
  });
  
  // Subscribe to theme store
  themeStore.subscribe(value => {
    theme = value;
  });
  
  function toggleTheme() {
    themeStore.toggleTheme();
  }
  
  function logout() {
    authStore.logout();
    navigate('/', { replace: true });
  }
  
  function toggleMenu() {
    isMenuOpen = !isMenuOpen;
  }
</script>

<nav class="bg-white dark:bg-gray-800 shadow-md">
  <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
    <div class="flex justify-between h-16">
      <div class="flex">
        <div class="flex-shrink-0 flex items-center">
          <Link to="/" class="text-xl font-bold text-primary-600 dark:text-primary-400">
            AstroAssistance
          </Link>
        </div>
      </div>
      
      <div class="flex items-center">
        <button 
          class="p-2 rounded-md text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
          on:click={toggleTheme}
          aria-label="Toggle theme"
        >
          {#if theme === 'dark'}
            <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
            </svg>
          {:else}
            <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
            </svg>
          {/if}
        </button>
        
        <div class="ml-4 relative">
          <button 
            class="flex items-center text-sm font-medium text-gray-700 hover:text-gray-900 dark:text-gray-300 dark:hover:text-white focus:outline-none"
            on:click={toggleMenu}
            aria-expanded={isMenuOpen}
          >
            <span class="mr-2">{user?.name || 'User'}</span>
            <svg class="h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
              <path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd" />
            </svg>
          </button>
          
          {#if isMenuOpen}
            <div 
              class="origin-top-right absolute right-0 mt-2 w-48 rounded-md shadow-lg bg-white dark:bg-gray-800 ring-1 ring-black ring-opacity-5 focus:outline-none"
              role="menu"
              aria-orientation="vertical"
              aria-labelledby="user-menu"
            >
              <div class="py-1" role="none">
                <Link to="/settings" class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-700" role="menuitem">
                  Settings
                </Link>
                <button 
                  class="block w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-700" 
                  role="menuitem"
                  on:click={logout}
                >
                  Sign out
                </button>
              </div>
            </div>
          {/if}
        </div>
      </div>
    </div>
  </div>
</nav>