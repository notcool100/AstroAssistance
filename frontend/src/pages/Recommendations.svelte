<script lang="ts">
  import { onMount } from 'svelte';
  
  let recommendations = [];
  let loading = true;
  let error = null;
  let activeCategory = 'all';
  
  // Function to fetch recommendations from the API
  async function fetchRecommendations() {
    loading = true;
    error = null;
    
    try {
      // Get the auth token from localStorage
      const token = localStorage.getItem('token');
      
      if (!token) {
        throw new Error('Authentication required');
      }
      
      // Fetch recommendations from the API
      const response = await fetch('http://localhost:8000/api/recommendations', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (!response.ok) {
        throw new Error('Failed to fetch recommendations');
      }
      
      const data = await response.json();
      
      // Transform API data to match our frontend format
      recommendations = data.recommendations.map(rec => ({
        id: rec.id,
        title: rec.content.split('.')[0], // Use the first sentence as title
        description: rec.content,
        category: rec.type,
        impact: rec.type.includes('risk') || rec.type.includes('warning') ? 'high' : 'medium',
        aiConfidence: rec.metadata?.confidence ? Math.round(rec.metadata.confidence * 100) : 85,
        implemented: rec.applied,
        createdAt: rec.createdAt
      }));
      
      loading = false;
    } catch (err) {
      console.error('Error fetching recommendations:', err);
      error = err.message;
      loading = false;
      
      // Fallback to placeholder data if API call fails
      recommendations = [
        {
          id: 1,
          title: 'Optimize your morning routine',
          description: 'Based on your activity patterns, starting your day 30 minutes earlier could improve your productivity by 15%.',
          category: 'productivity',
          impact: 'high',
          aiConfidence: 85,
          implemented: false,
          createdAt: '2025-07-28'
        },
        {
          id: 2,
          title: 'Schedule focused work blocks',
          description: 'Your data shows you\'re most productive between 10am-12pm. Consider blocking this time for your most important tasks.',
          category: 'productivity',
          impact: 'medium',
          aiConfidence: 92,
          implemented: true,
          createdAt: '2025-07-25'
        },
        {
          id: 3,
          title: 'Take more frequent short breaks',
          description: 'Your productivity decreases after 90 minutes of continuous work. Try implementing the Pomodoro technique with 5-minute breaks.',
          category: 'wellbeing',
          impact: 'medium',
          aiConfidence: 78,
          implemented: false,
          createdAt: '2025-07-22'
        },
        {
          id: 4,
          title: 'Delegate routine administrative tasks',
          description: 'You spend approximately 5 hours weekly on low-value administrative tasks that could potentially be delegated.',
          category: 'efficiency',
          impact: 'high',
          aiConfidence: 88,
          implemented: false,
          createdAt: '2025-07-20'
        },
        {
          id: 5,
          title: 'Reduce meeting duration',
          description: 'Analysis shows your 60-minute meetings could be just as effective at 45 minutes, saving you 3+ hours weekly.',
          category: 'efficiency',
          impact: 'medium',
          aiConfidence: 76,
          implemented: false,
          createdAt: '2025-07-18'
        }
      ];
    }
  }
  
  // Function to generate new recommendations
  async function generateRecommendations() {
    loading = true;
    error = null;
    
    try {
      // Get the auth token from localStorage
      const token = localStorage.getItem('token');
      
      if (!token) {
        throw new Error('Authentication required');
      }
      
      // Call the API to generate new recommendations
      const response = await fetch('http://localhost:8000/api/recommendations/generate?count=3', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (!response.ok) {
        throw new Error('Failed to generate recommendations');
      }
      
      // Fetch the updated list of recommendations
      await fetchRecommendations();
      
      // Show success message
      alert('New AI recommendations generated successfully!');
    } catch (err) {
      console.error('Error generating recommendations:', err);
      error = err.message;
      loading = false;
    }
  }
  
  // Function to submit feedback on a recommendation
  async function submitFeedback(id, rating, comment = '') {
    try {
      // Get the auth token from localStorage
      const token = localStorage.getItem('token');
      
      if (!token) {
        throw new Error('Authentication required');
      }
      
      // Submit feedback to the API
      const response = await fetch('http://localhost:8000/api/ai/feedback', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          recommendationId: id,
          rating,
          comment
        })
      });
      
      if (!response.ok) {
        throw new Error('Failed to submit feedback');
      }
      
      // Show success message
      alert('Thank you for your feedback! This helps improve our AI recommendations.');
    } catch (err) {
      console.error('Error submitting feedback:', err);
      alert(`Error: ${err.message}`);
    }
  }
  
  onMount(async () => {
    // Fetch recommendations when the component mounts
    await fetchRecommendations();
  });
  
  function setCategory(category: string) {
    activeCategory = category;
  }
  
  function toggleImplemented(id: number) {
    recommendations = recommendations.map(rec => 
      rec.id === id 
        ? { ...rec, implemented: !rec.implemented } 
        : rec
    );
  }
  
  function dismissRecommendation(id: number) {
    recommendations = recommendations.filter(rec => rec.id !== id);
  }
  
  function getImpactClass(impact: string): string {
    switch (impact.toLowerCase()) {
      case 'high': return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
      case 'medium': return 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200';
      case 'low': return 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200';
      default: return 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200';
    }
  }
  
  function getCategoryClass(category: string): string {
    switch (category.toLowerCase()) {
      case 'productivity': return 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200';
      case 'efficiency': return 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200';
      case 'wellbeing': return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
      default: return 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200';
    }
  }
  
  function formatDate(dateString: string): string {
    return new Date(dateString).toLocaleDateString();
  }
  
  $: filteredRecommendations = activeCategory === 'all' 
    ? recommendations 
    : recommendations.filter(rec => rec.category === activeCategory);
</script>

<div class="container mx-auto">
  <div class="mb-6">
    <h1 class="text-2xl font-bold mb-4">AI Recommendations</h1>
    <p class="text-gray-600 dark:text-gray-300">
      Personalized suggestions based on your activity patterns and productivity data.
    </p>
  </div>
  
  <div class="mb-6 flex justify-between items-center">
    <div class="flex flex-wrap gap-2">
      <button 
        on:click={() => setCategory('all')}
        class={`px-4 py-2 rounded-md ${activeCategory === 'all' 
          ? 'bg-blue-600 text-white' 
          : 'bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200'}`}
      >
        All
      </button>
      <button 
        on:click={() => setCategory('task')}
        class={`px-4 py-2 rounded-md ${activeCategory === 'task' 
          ? 'bg-blue-600 text-white' 
          : 'bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200'}`}
      >
        Tasks
      </button>
      <button 
        on:click={() => setCategory('goal')}
        class={`px-4 py-2 rounded-md ${activeCategory === 'goal' 
          ? 'bg-blue-600 text-white' 
          : 'bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200'}`}
      >
        Goals
      </button>
      <button 
        on:click={() => setCategory('productivity')}
        class={`px-4 py-2 rounded-md ${activeCategory === 'productivity' 
          ? 'bg-blue-600 text-white' 
          : 'bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200'}`}
      >
        Productivity
      </button>
      <button 
        on:click={() => setCategory('wellbeing')}
        class={`px-4 py-2 rounded-md ${activeCategory === 'wellbeing' 
          ? 'bg-blue-600 text-white' 
          : 'bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200'}`}
      >
        Wellbeing
      </button>
    </div>
    
    <button 
      on:click={generateRecommendations}
      class="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-md flex items-center"
      disabled={loading}
    >
      {#if loading}
        <div class="animate-spin rounded-full h-4 w-4 border-t-2 border-b-2 border-white mr-2"></div>
      {/if}
      Generate New Recommendations
    </button>
  </div>
  
  {#if loading}
    <div class="flex justify-center items-center h-64">
      <div class="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
    </div>
  {:else if error}
    <div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
      <p>Error loading recommendations: {error}</p>
    </div>
  {:else if filteredRecommendations.length === 0}
    <div class="bg-gray-100 dark:bg-gray-800 p-6 rounded-lg text-center">
      <p class="text-lg">No recommendations available in this category.</p>
      <p class="mt-2">Check back later as we analyze more of your activity data.</p>
    </div>
  {:else}
    <div class="space-y-6">
      {#each filteredRecommendations as recommendation}
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
          <div class="flex justify-between items-start">
            <div>
              <h2 class="text-xl font-semibold mb-2">{recommendation.title}</h2>
              <p class="text-gray-600 dark:text-gray-300 mb-4">{recommendation.description}</p>
              
              <div class="flex flex-wrap gap-2 mb-4">
                <span class={`text-xs px-2 py-1 rounded-full ${getCategoryClass(recommendation.category)}`}>
                  {recommendation.category.charAt(0).toUpperCase() + recommendation.category.slice(1)}
                </span>
                <span class={`text-xs px-2 py-1 rounded-full ${getImpactClass(recommendation.impact)}`}>
                  {recommendation.impact.charAt(0).toUpperCase() + recommendation.impact.slice(1)} Impact
                </span>
                <span class="text-xs px-2 py-1 rounded-full bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200">
                  AI Confidence: {recommendation.aiConfidence}%
                </span>
              </div>
              
              <div class="text-sm text-gray-500 dark:text-gray-400">
                Suggested on {formatDate(recommendation.createdAt)}
              </div>
            </div>
            
            <div class="flex space-x-2">
              <div class="flex flex-col space-y-2">
                <button 
                  on:click={() => toggleImplemented(recommendation.id)}
                  class={`px-3 py-1 rounded-md ${recommendation.implemented 
                    ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' 
                    : 'bg-blue-600 hover:bg-blue-700 text-white'}`}
                >
                  {recommendation.implemented ? 'Implemented' : 'Implement'}
                </button>
                <div class="flex space-x-2">
                  <button 
                    on:click={() => submitFeedback(recommendation.id, 5)}
                    class="px-2 py-1 rounded-md bg-green-100 hover:bg-green-200 text-green-800 dark:bg-green-900 dark:hover:bg-green-800 dark:text-green-200"
                    title="This recommendation is helpful"
                  >
                    👍
                  </button>
                  <button 
                    on:click={() => submitFeedback(recommendation.id, 1)}
                    class="px-2 py-1 rounded-md bg-red-100 hover:bg-red-200 text-red-800 dark:bg-red-900 dark:hover:bg-red-800 dark:text-red-200"
                    title="This recommendation is not helpful"
                  >
                    👎
                  </button>
                  <button 
                    on:click={() => dismissRecommendation(recommendation.id)}
                    class="px-2 py-1 rounded-md bg-gray-200 hover:bg-gray-300 text-gray-800 dark:bg-gray-700 dark:hover:bg-gray-600 dark:text-gray-200"
                    title="Dismiss this recommendation"
                  >
                    ✕
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      {/each}
    </div>
  {/if}
</div>