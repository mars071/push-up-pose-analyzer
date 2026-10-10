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

/**
 * Publishes the intrinsic aspect ratio of a webcam stream as the
 * `--video-aspect` CSS custom property on the video's `.video-wrapper`.
 *
 * The value is a plain number (width / height) so the stylesheet can use
 * it both in `aspect-ratio` and inside `calc()` to derive a width.
 *
 * Phone cameras deliver portrait streams in portrait orientation; the
 * mobile stylesheet uses this variable to size the preview box to the
 * stream instead of forcing a 4:3 landscape pillarbox.
 */
export function trackVideoAspect(video: HTMLVideoElement) {
  const wrapper = video.closest<HTMLElement>('.video-wrapper');
  if (!wrapper) return;

  const apply = () => {
    if (video.videoWidth > 0 && video.videoHeight > 0) {
      wrapper.style.setProperty('--video-aspect', (video.videoWidth / video.videoHeight).toFixed(4));
    }
  };

  apply();
  video.addEventListener('loadedmetadata', apply);
  // iOS re-reports dimensions when the device rotates mid-stream.
  video.addEventListener('resize', apply);
}

/** Clears the aspect ratio published by {@link trackVideoAspect}. */
export function clearVideoAspect(video: HTMLVideoElement) {
  video.closest<HTMLElement>('.video-wrapper')?.style.removeProperty('--video-aspect');
}
