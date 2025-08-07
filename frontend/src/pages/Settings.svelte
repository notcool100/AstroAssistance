<script lang="ts">
  import { onMount } from 'svelte';
  import { themeStore } from '../stores/theme';
  import { authStore } from '../stores/auth';
  
  let user = {
    name: 'Alex Johnson',
    email: 'alex@example.com',
    profileImage: null
  };
  
  let theme: string;
  let notificationSettings = {
    email: true,
    push: true,
    taskReminders: true,
    goalUpdates: true,
    weeklyReports: true
  };
  
  let privacySettings = {
    shareActivityData: true,
    allowAnonymousDataCollection: true
  };
  
  let integrations = [
    { id: 'google_calendar', name: 'Google Calendar', connected: true, icon: 'calendar' },
    { id: 'slack', name: 'Slack', connected: false, icon: 'chat' },
    { id: 'github', name: 'GitHub', connected: true, icon: 'code' },
    { id: 'trello', name: 'Trello', connected: false, icon: 'board' }
  ];
  
  let activeTab = 'profile';
  let passwordForm = {
    currentPassword: '',
    newPassword: '',
    confirmPassword: ''
  };
  let showSuccessMessage = false;
  let successMessage = '';
  
  // Subscribe to theme store
  themeStore.subscribe(value => {
    theme = value;
  });
  
  function setTheme(newTheme: string) {
    themeStore.setTheme(newTheme);
  }
  
  function setActiveTab(tab: string) {
    activeTab = tab;
  }
  
  function saveProfile() {
    // In a real app, this would be an API call
    // await fetch('/api/user/profile', {
    //   method: 'PUT',
    //   headers: { 'Content-Type': 'application/json' },
    //   body: JSON.stringify(user)
    // });
    
    showSuccess('Profile updated successfully');
  }
  
  function saveNotifications() {
    // In a real app, this would be an API call
    // await fetch('/api/user/notifications', {
    //   method: 'PUT',
    //   headers: { 'Content-Type': 'application/json' },
    //   body: JSON.stringify(notificationSettings)
    // });
    
    showSuccess('Notification preferences saved');
  }
  
  function savePrivacy() {
    // In a real app, this would be an API call
    // await fetch('/api/user/privacy', {
    //   method: 'PUT',
    //   headers: { 'Content-Type': 'application/json' },
    //   body: JSON.stringify(privacySettings)
    // });
    
    showSuccess('Privacy settings updated');
  }
  
  function changePassword() {
    if (passwordForm.newPassword !== passwordForm.confirmPassword) {
      // Show error
      return;
    }
    
    // In a real app, this would be an API call
    // await fetch('/api/user/password', {
    //   method: 'PUT',
    //   headers: { 'Content-Type': 'application/json' },
    //   body: JSON.stringify({
    //     currentPassword: passwordForm.currentPassword,
    //     newPassword: passwordForm.newPassword
    //   })
    // });
    
    passwordForm = {
      currentPassword: '',
      newPassword: '',
      confirmPassword: ''
    };
    
    showSuccess('Password changed successfully');
  }
  
  function toggleIntegration(id: string) {
    integrations = integrations.map(integration => 
      integration.id === id 
        ? { ...integration, connected: !integration.connected } 
        : integration
    );
    
    const integration = integrations.find(i => i.id === id);
    if (integration) {
      showSuccess(`${integration.name} ${integration.connected ? 'connected' : 'disconnected'} successfully`);
    }
  }
  
  function showSuccess(message: string) {
    successMessage = message;
    showSuccessMessage = true;
    setTimeout(() => {
      showSuccessMessage = false;
    }, 3000);
  }
  
  function logout() {
    // Clear auth store
    authStore.logout();
    // In a real app, navigate to login page
  }
  
  function deleteAccount() {
    if (confirm('Are you sure you want to delete your account? This action cannot be undone.')) {
      // In a real app, this would be an API call
      // await fetch('/api/user', {
      //   method: 'DELETE'
      // });
      
      // Clear auth store
      authStore.logout();
      // In a real app, navigate to login page
    }
  }
</script>

<div class="container mx-auto">
  <h1 class="text-2xl font-bold mb-6">Settings</h1>
  
  {#if showSuccessMessage}
    <div class="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded mb-6 flex justify-between items-center">
      <p>{successMessage}</p>
      <button on:click={() => showSuccessMessage = false} class="text-green-700">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
        </svg>
      </button>
    </div>
  {/if}
  
  <div class="flex flex-col md:flex-row gap-6">
    <!-- Sidebar -->
    <div class="w-full md:w-64 bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
      <nav class="space-y-1">
        <button 
          on:click={() => setActiveTab('profile')}
          class={`w-full text-left px-4 py-2 rounded-md flex items-center ${activeTab === 'profile' ? 'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-200' : 'hover:bg-gray-100 dark:hover:bg-gray-700'}`}
        >
          <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path>
          </svg>
          Profile
        </button>
        
        <button 
          on:click={() => setActiveTab('appearance')}
          class={`w-full text-left px-4 py-2 rounded-md flex items-center ${activeTab === 'appearance' ? 'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-200' : 'hover:bg-gray-100 dark:hover:bg-gray-700'}`}
        >
          <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01"></path>
          </svg>
          Appearance
        </button>
        
        <button 
          on:click={() => setActiveTab('notifications')}
          class={`w-full text-left px-4 py-2 rounded-md flex items-center ${activeTab === 'notifications' ? 'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-200' : 'hover:bg-gray-100 dark:hover:bg-gray-700'}`}
        >
          <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9"></path>
          </svg>
          Notifications
        </button>
        
        <button 
          on:click={() => setActiveTab('privacy')}
          class={`w-full text-left px-4 py-2 rounded-md flex items-center ${activeTab === 'privacy' ? 'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-200' : 'hover:bg-gray-100 dark:hover:bg-gray-700'}`}
        >
          <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"></path>
          </svg>
          Privacy & Security
        </button>
        
        <button 
          on:click={() => setActiveTab('integrations')}
          class={`w-full text-left px-4 py-2 rounded-md flex items-center ${activeTab === 'integrations' ? 'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-200' : 'hover:bg-gray-100 dark:hover:bg-gray-700'}`}
        >
          <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
          </svg>
          Integrations
        </button>
        
        <button 
          on:click={() => setActiveTab('account')}
          class={`w-full text-left px-4 py-2 rounded-md flex items-center ${activeTab === 'account' ? 'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-200' : 'hover:bg-gray-100 dark:hover:bg-gray-700'}`}
        >
          <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"></path>
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path>
          </svg>
          Account
        </button>
      </nav>
    </div>
    
    <!-- Content -->
    <div class="flex-1 bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
      {#if activeTab === 'profile'}
        <h2 class="text-xl font-semibold mb-4">Profile Settings</h2>
        <form on:submit|preventDefault={saveProfile} class="space-y-4">
          <div>
            <label for="name" class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Name</label>
            <input 
              type="text" 
              id="name" 
              bind:value={user.name} 
              class="w-full px-3 py-2 border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
            />
          </div>
          
          <div>
            <label for="email" class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Email</label>
            <input 
              type="email" 
              id="email" 
              bind:value={user.email} 
              class="w-full px-3 py-2 border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
            />
          </div>
          
          <div>
            <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Profile Picture</label>
            <div class="flex items-center space-x-4">
              <div class="w-16 h-16 bg-gray-200 dark:bg-gray-700 rounded-full flex items-center justify-center">
                {#if user.profileImage}
                  <img src={user.profileImage} alt="Profile" class="w-16 h-16 rounded-full object-cover" />
                {:else}
                  <svg class="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path>
                  </svg>
                {/if}
              </div>
              <button type="button" class="px-3 py-1 bg-gray-200 dark:bg-gray-700 rounded-md hover:bg-gray-300 dark:hover:bg-gray-600">
                Upload Image
              </button>
            </div>
          </div>
          
          <div class="pt-4">
            <button type="submit" class="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md">
              Save Changes
            </button>
          </div>
        </form>
      {:else if activeTab === 'appearance'}
        <h2 class="text-xl font-semibold mb-4">Appearance Settings</h2>
        <div class="space-y-6">
          <div>
            <h3 class="text-lg font-medium mb-2">Theme</h3>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
              <button 
                on:click={() => setTheme('light')}
                class={`p-4 border rounded-md flex flex-col items-center ${theme === 'light' ? 'border-blue-500 bg-blue-50 dark:bg-blue-900' : 'border-gray-300 dark:border-gray-700'}`}
              >
                <div class="w-full h-24 bg-white border border-gray-200 rounded-md mb-2 flex flex-col">
                  <div class="h-6 bg-gray-100 border-b border-gray-200"></div>
                  <div class="flex flex-1">
                    <div class="w-1/4 bg-gray-50 border-r border-gray-200"></div>
                    <div class="w-3/4 p-2">
                      <div class="w-3/4 h-3 bg-gray-200 rounded mb-2"></div>
                      <div class="w-1/2 h-3 bg-gray-200 rounded"></div>
                    </div>
                  </div>
                </div>
                <span class={theme === 'light' ? 'text-blue-600 dark:text-blue-400 font-medium' : ''}>Light</span>
              </button>
              
              <button 
                on:click={() => setTheme('dark')}
                class={`p-4 border rounded-md flex flex-col items-center ${theme === 'dark' ? 'border-blue-500 bg-blue-50 dark:bg-blue-900' : 'border-gray-300 dark:border-gray-700'}`}
              >
                <div class="w-full h-24 bg-gray-900 border border-gray-700 rounded-md mb-2 flex flex-col">
                  <div class="h-6 bg-gray-800 border-b border-gray-700"></div>
                  <div class="flex flex-1">
                    <div class="w-1/4 bg-gray-800 border-r border-gray-700"></div>
                    <div class="w-3/4 p-2">
                      <div class="w-3/4 h-3 bg-gray-700 rounded mb-2"></div>
                      <div class="w-1/2 h-3 bg-gray-700 rounded"></div>
                    </div>
                  </div>
                </div>
                <span class={theme === 'dark' ? 'text-blue-600 dark:text-blue-400 font-medium' : ''}>Dark</span>
              </button>
              
              <button 
                on:click={() => setTheme('system')}
                class={`p-4 border rounded-md flex flex-col items-center ${theme === 'system' ? 'border-blue-500 bg-blue-50 dark:bg-blue-900' : 'border-gray-300 dark:border-gray-700'}`}
              >
                <div class="w-full h-24 bg-gradient-to-r from-white to-gray-900 border border-gray-300 dark:border-gray-700 rounded-md mb-2 flex items-center justify-center">
                  <svg class="w-10 h-10 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"></path>
                  </svg>
                </div>
                <span class={theme === 'system' ? 'text-blue-600 dark:text-blue-400 font-medium' : ''}>System</span>
              </button>
            </div>
          </div>
        </div>
      {:else if activeTab === 'notifications'}
        <h2 class="text-xl font-semibold mb-4">Notification Settings</h2>
        <form on:submit|preventDefault={saveNotifications} class="space-y-4">
          <div class="space-y-3">
            <div class="flex items-center justify-between">
              <div>
                <h3 class="font-medium">Email Notifications</h3>
                <p class="text-sm text-gray-500 dark:text-gray-400">Receive notifications via email</p>
              </div>
              <label class="relative inline-flex items-center cursor-pointer">
                <input type="checkbox" bind:checked={notificationSettings.email} class="sr-only peer">
                <div class="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
              </label>
            </div>
            
            <div class="flex items-center justify-between">
              <div>
                <h3 class="font-medium">Push Notifications</h3>
                <p class="text-sm text-gray-500 dark:text-gray-400">Receive notifications in your browser</p>
              </div>
              <label class="relative inline-flex items-center cursor-pointer">
                <input type="checkbox" bind:checked={notificationSettings.push} class="sr-only peer">
                <div class="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
              </label>
            </div>
            
            <hr class="my-4 border-gray-200 dark:border-gray-700">
            
            <div class="flex items-center justify-between">
              <div>
                <h3 class="font-medium">Task Reminders</h3>
                <p class="text-sm text-gray-500 dark:text-gray-400">Get notified about upcoming tasks</p>
              </div>
              <label class="relative inline-flex items-center cursor-pointer">
                <input type="checkbox" bind:checked={notificationSettings.taskReminders} class="sr-only peer">
                <div class="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
              </label>
            </div>
            
            <div class="flex items-center justify-between">
              <div>
                <h3 class="font-medium">Goal Updates</h3>
                <p class="text-sm text-gray-500 dark:text-gray-400">Get notified about goal progress</p>
              </div>
              <label class="relative inline-flex items-center cursor-pointer">
                <input type="checkbox" bind:checked={notificationSettings.goalUpdates} class="sr-only peer">
                <div class="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
              </label>
            </div>
            
            <div class="flex items-center justify-between">
              <div>
                <h3 class="font-medium">Weekly Reports</h3>
                <p class="text-sm text-gray-500 dark:text-gray-400">Receive weekly productivity reports</p>
              </div>
              <label class="relative inline-flex items-center cursor-pointer">
                <input type="checkbox" bind:checked={notificationSettings.weeklyReports} class="sr-only peer">
                <div class="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
              </label>
            </div>
          </div>
          
          <div class="pt-4">
            <button type="submit" class="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md">
              Save Preferences
            </button>
          </div>
        </form>
      {:else if activeTab === 'privacy'}
        <h2 class="text-xl font-semibold mb-4">Privacy & Security</h2>
        <div class="space-y-6">
          <div>
            <h3 class="text-lg font-medium mb-3">Privacy Settings</h3>
            <form on:submit|preventDefault={savePrivacy} class="space-y-4">
              <div class="space-y-3">
                <div class="flex items-center justify-between">
                  <div>
                    <h4 class="font-medium">Share Activity Data</h4>
                    <p class="text-sm text-gray-500 dark:text-gray-400">Allow the app to use your activity data for personalized recommendations</p>
                  </div>
                  <label class="relative inline-flex items-center cursor-pointer">
                    <input type="checkbox" bind:checked={privacySettings.shareActivityData} class="sr-only peer">
                    <div class="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
                  </label>
                </div>
                
                <div class="flex items-center justify-between">
                  <div>
                    <h4 class="font-medium">Anonymous Data Collection</h4>
                    <p class="text-sm text-gray-500 dark:text-gray-400">Allow anonymous usage data to be collected for app improvement</p>
                  </div>
                  <label class="relative inline-flex items-center cursor-pointer">
                    <input type="checkbox" bind:checked={privacySettings.allowAnonymousDataCollection} class="sr-only peer">
                    <div class="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
                  </label>
                </div>
              </div>
              
              <div class="pt-2">
                <button type="submit" class="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md">
                  Save Privacy Settings
                </button>
              </div>
            </form>
          </div>
          
          <hr class="my-6 border-gray-200 dark:border-gray-700">
          
          <div>
            <h3 class="text-lg font-medium mb-3">Change Password</h3>
            <form on:submit|preventDefault={changePassword} class="space-y-4">
              <div>
                <label for="currentPassword" class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Current Password</label>
                <input 
                  type="password" 
                  id="currentPassword" 
                  bind:value={passwordForm.currentPassword} 
                  required
                  class="w-full px-3 py-2 border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
                />
              </div>
              
              <div>
                <label for="newPassword" class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">New Password</label>
                <input 
                  type="password" 
                  id="newPassword" 
                  bind:value={passwordForm.newPassword} 
                  required
                  class="w-full px-3 py-2 border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
                />
              </div>
              
              <div>
                <label for="confirmPassword" class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Confirm New Password</label>
                <input 
                  type="password" 
                  id="confirmPassword" 
                  bind:value={passwordForm.confirmPassword} 
                  required
                  class="w-full px-3 py-2 border border-gray-300 dark:border-gray-700 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
                />
              </div>
              
              <div class="pt-2">
                <button type="submit" class="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md">
                  Change Password
                </button>
              </div>
            </form>
          </div>
        </div>
      {:else if activeTab === 'integrations'}
        <h2 class="text-xl font-semibold mb-4">Integrations</h2>
        <p class="text-gray-600 dark:text-gray-300 mb-6">
          Connect AstroAssistance with your favorite tools and services.
        </p>
        
        <div class="space-y-4">
          {#each integrations as integration}
            <div class="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <div class="flex items-center">
                <div class="w-10 h-10 bg-blue-100 dark:bg-blue-900 rounded-full flex items-center justify-center mr-4">
                  {#if integration.icon === 'calendar'}
                    <svg class="w-6 h-6 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
                    </svg>
                  {:else if integration.icon === 'chat'}
                    <svg class="w-6 h-6 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"></path>
                    </svg>
                  {:else if integration.icon === 'code'}
                    <svg class="w-6 h-6 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"></path>
                    </svg>
                  {:else if integration.icon === 'board'}
                    <svg class="w-6 h-6 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"></path>
                    </svg>
                  {/if}
                </div>
                <div>
                  <h3 class="font-medium">{integration.name}</h3>
                  <p class="text-sm text-gray-500 dark:text-gray-400">
                    {integration.connected ? 'Connected' : 'Not connected'}
                  </p>
                </div>
              </div>
              <button 
                on:click={() => toggleIntegration(integration.id)}
                class={integration.connected 
                  ? 'px-3 py-1 bg-red-100 text-red-700 hover:bg-red-200 dark:bg-red-900 dark:text-red-200 dark:hover:bg-red-800 rounded-md' 
                  : 'px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white rounded-md'}
              >
                {integration.connected ? 'Disconnect' : 'Connect'}
              </button>
            </div>
          {/each}
        </div>
      {:else if activeTab === 'account'}
        <h2 class="text-xl font-semibold mb-4">Account Settings</h2>
        <div class="space-y-6">
          <div>
            <h3 class="text-lg font-medium mb-3">Account Actions</h3>
            <div class="space-y-4">
              <div>
                <button 
                  on:click={logout}
                  class="w-full md:w-auto px-4 py-2 bg-gray-200 hover:bg-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600 rounded-md flex items-center justify-center"
                >
                  <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"></path>
                  </svg>
                  Log Out
                </button>
              </div>
              
              <hr class="my-6 border-gray-200 dark:border-gray-700">
              
              <div>
                <h3 class="text-lg font-medium text-red-600 dark:text-red-400 mb-2">Danger Zone</h3>
                <p class="text-gray-600 dark:text-gray-300 mb-4">
                  Once you delete your account, there is no going back. Please be certain.
                </p>
                <button 
                  on:click={deleteAccount}
                  class="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-md"
                >
                  Delete Account
                </button>
              </div>
            </div>
          </div>
        </div>
      {/if}
    </div>
  </div>
</div>