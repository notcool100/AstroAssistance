<script lang="ts">
  import { Link, useLocation } from 'svelte-navigator';
  
  const location = useLocation();
  let currentPath: string;
  
  // Subscribe to location changes
  location.subscribe(value => {
    currentPath = value.pathname;
  });
  
  // Navigation items
  const navItems = [
    { path: '/', label: 'Dashboard', icon: 'home' },
    { path: '/tasks', label: 'Tasks', icon: 'task' },
    { path: '/goals', label: 'Goals', icon: 'goal' },
    { path: '/reminders', label: 'Reminders', icon: 'reminder' },
    { path: '/recommendations', label: 'Recommendations', icon: 'recommendation' },
  ];
  
  // Icon components
  const icons = {
    home: `<svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
    </svg>`,
    task: `<svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
    </svg>`,
    goal: `<svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
    </svg>`,
    reminder: `<svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
    </svg>`,
    recommendation: `<svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
    </svg>`,
  };
  
  function isActive(path: string): boolean {
    if (path === '/') {
      return currentPath === '/';
    }
    return currentPath.startsWith(path);
  }
</script>

<aside class="w-64 bg-white dark:bg-gray-800 shadow-md hidden md:block">
  <div class="h-full px-3 py-4 overflow-y-auto">
    <ul class="space-y-2">
      {#each navItems as item}
        <li>
          <Link 
            to={item.path} 
            class="flex items-center p-2 text-base font-normal rounded-lg {isActive(item.path) ? 'bg-primary-100 text-primary-900 dark:bg-primary-900 dark:text-primary-100' : 'text-gray-900 hover:bg-gray-100 dark:text-white dark:hover:bg-gray-700'}"
          >
            <div class="w-6 h-6 transition duration-75 {isActive(item.path) ? 'text-primary-600 dark:text-primary-400' : 'text-gray-500 dark:text-gray-400'}">
              {@html icons[item.icon]}
            </div>
            <span class="ml-3">{item.label}</span>
          </Link>
        </li>
      {/each}
    </ul>
  </div>
</aside>

<!-- Mobile menu -->
<div class="md:hidden fixed bottom-0 left-0 z-50 w-full h-16 bg-white border-t border-gray-200 dark:bg-gray-800 dark:border-gray-700">
  <div class="grid h-full grid-cols-5">
    {#each navItems as item}
      <Link 
        to={item.path} 
        class="inline-flex flex-col items-center justify-center px-5 hover:bg-gray-50 dark:hover:bg-gray-700 {isActive(item.path) ? 'text-primary-600 dark:text-primary-400' : 'text-gray-500 dark:text-gray-400'}"
      >
        <div class="w-6 h-6">
          {@html icons[item.icon]}
        </div>
        <span class="text-xs mt-1">{item.label}</span>
      </Link>
    {/each}
  </div>
</div>