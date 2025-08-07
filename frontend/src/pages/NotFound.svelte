<script lang="ts">
  import { Link } from 'svelte-navigator';
</script>

<div class="min-h-[70vh] flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
  <div class="max-w-md w-full text-center">
    <svg xmlns="http://www.w3.org/2000/svg" class="h-24 w-24 mx-auto text-gray-400 dark:text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
    <h1 class="mt-4 text-3xl font-bold text-gray-900 dark:text-white">Page not found</h1>
    <p class="mt-2 text-base text-gray-500 dark:text-gray-400">Sorry, we couldn't find the page you're looking for.</p>
    <div class="mt-6">
      <Link to="/" class="btn btn-primary">Go back home</Link>
    </div>
  </div>
</div>