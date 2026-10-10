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

export function renderSidebar(container: HTMLElement) {
  container.innerHTML = `
    <div class="sidebar-header">
      <button class="menu-toggle material-icons" style="margin-right: 24px; color: var(--text-secondary); background: none; border: none; font-size: 24px; cursor: pointer;">menu_open</button>
    </div>
    <nav class="sidebar-nav">
      <ul>
        <li><a href="#/vision/swimming_analyzer" class="nav-button nav-vision" data-task="swimming-analyzer">Push Up Pose Analyzer</a></li>
      </ul>
    </nav>
    <div class="sidebar-footer">
      <a href="https://goo.gle/mediapipe-privacy" target="_blank" rel="noopener noreferrer" class="sidebar-footer-link">
        <span class="material-icons">privacy_tip</span>
        <span>Privacy Notice</span>
      </a>
    </div>
  `;
}
