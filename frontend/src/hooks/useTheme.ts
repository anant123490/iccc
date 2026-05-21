import { useEffect, useState } from 'react';
import { getSettings, saveSettings } from '../services/storage';

export function useTheme() {
  const [darkMode, setDarkMode] = useState(false);

  useEffect(() => {
    getSettings().then((s) => {
      setDarkMode(s.darkMode);
      document.documentElement.classList.toggle('dark', s.darkMode);
    });
  }, []);

  const toggle = async () => {
    const settings = await getSettings();
    const next = !settings.darkMode;
    settings.darkMode = next;
    await saveSettings(settings);
    setDarkMode(next);
    document.documentElement.classList.toggle('dark', next);
  };

  return { darkMode, toggle };
}
