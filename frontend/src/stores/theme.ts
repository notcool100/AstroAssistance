import { writable } from 'svelte/store';

function createThemeStore() {
  const { subscribe, set, update } = writable<string>('light');

  return {
    subscribe,
    setTheme: (theme: string) => {
      localStorage.setItem('theme', theme);
      set(theme);
    },
    toggleTheme: () => {
      update(theme => {
        const newTheme = theme === 'light' ? 'dark' : 'light';
        localStorage.setItem('theme', newTheme);
        return newTheme;
      });
    },
  };
}

export const themeStore = createThemeStore();