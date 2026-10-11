/**
 * Copyright 2026 The MediaPipe Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import './app_clean.css';
import { setupSwimmingAnalyzer, cleanupSwimmingAnalyzer } from './tasks/swimming-analyzer';

import { renderSidebar } from './ui/sidebar';
import { renderMobileNav } from './ui/mobile-nav';

const app = document.querySelector<HTMLDivElement>('#app')!;

// 1. Setup App Shell
app.innerHTML = `
  <div class="app-container">
    <aside class="sidebar"></aside>
    <div class="sidebar-backdrop" aria-hidden="true"></div>
    <div class="mobile-header">
       <button class="menu-toggle material-icons" aria-label="Open navigation" style="margin-right: 24px; color: var(--text-secondary); background: none; border: none; font-size: 24px; cursor: pointer;">menu</button>
       <div id="mobile-nav-container" style="display: flex; align-items: center; flex-grow: 1; min-width: 0;"></div>
    </div>
    <main class="main-content">
      <div id="task-stage" class="task-stage"></div>
    </main>
  </div>
`;

// 2. Render Global Components
const sidebar = app.querySelector('.sidebar') as HTMLElement;
renderSidebar(sidebar);

const mobileNavContainer = app.querySelector('#mobile-nav-container') as HTMLElement;
renderMobileNav(mobileNavContainer);

// 3. Setup Navigation Logic (off-canvas drawer on small screens)
const sidebarBackdrop = app.querySelector('.sidebar-backdrop') as HTMLElement;

function setSidebarOpen(open: boolean) {
  sidebar.classList.toggle('open', open);
  sidebarBackdrop.classList.toggle('visible', open);
  document.body.classList.toggle('sidebar-open', open);
}

const menuToggles = app.querySelectorAll('.menu-toggle');
menuToggles.forEach((toggle) => {
  toggle.addEventListener('click', () => {
    setSidebarOpen(!sidebar.classList.contains('open'));
  });
});

sidebarBackdrop.addEventListener('click', () => setSidebarOpen(false));

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && sidebar.classList.contains('open')) setSidebarOpen(false);
});

// Close sidebar when a link is clicked
sidebar.addEventListener('click', (e) => {
  if ((e.target as HTMLElement).closest('a')) {
    setSidebarOpen(false);
  }
});

const mainContent = app.querySelector('#task-stage') as HTMLElement;

// 4. Router Setup
const routes = {
  '/vision/swimming_analyzer': {
    setup: setupSwimmingAnalyzer,
    cleanup: cleanupSwimmingAnalyzer,
    label: 'Push Up Pose Analyzer',
  },
};

let currentCleanup: (() => void) | undefined;

async function applyRoute() {
  const hash = window.location.hash.slice(1);

  // Handle root or invalid routes by defaulting to the analyzer. Changing
  // the hash fires `hashchange`, which routes again with the valid hash.
  if (!hash || !routes[hash as keyof typeof routes]) {
    window.location.hash = '/vision/swimming_analyzer';
    return;
  }

  const route = routes[hash as keyof typeof routes];

  // Cleanup previous task
  if (currentCleanup) {
    currentCleanup();
    currentCleanup = undefined;
  }

  // Clear main content area only
  mainContent.innerHTML = '';

  // Setup new task
  await route.setup(mainContent);
  currentCleanup = route.cleanup;
  document.title = route.label;

  // Update active state in sidebar
  const links = sidebar.querySelectorAll('a');
  links.forEach((l) => {
    if (l.getAttribute('href') === `#${hash}`) l.classList.add('active');
    else l.classList.remove('active');
  });
}

// Route changes are applied one at a time: a task's (async) setup must finish
// before it can be cleaned up, otherwise two instances – and two workers
// loading the same model – would be left running.
let routing: Promise<void> = Promise.resolve();
function router() {
  routing = routing.then(applyRoute).catch((err) => console.error('Failed to set up route:', err));
  return routing;
}

window.addEventListener('hashchange', router);

// Initialize router immediately to handle initial load
router();

// Expose cleanup method for testing environments to prevent leaks between tests
(window as any).cleanupActiveTask = () => {
  if (currentCleanup) {
    currentCleanup();
    currentCleanup = undefined;
  }
};
