/**
 * DeepGuard X — Dashboard scripts and telemetry visualizations
 */

document.addEventListener("DOMContentLoaded", function () {
  // Animate progress bars smoothly on load
  const progressBars = document.querySelectorAll(".progress-fill");
  progressBars.forEach(function (bar) {
    const targetWidth = bar.getAttribute("data-width");
    if (targetWidth) {
      setTimeout(function () {
        bar.style.width = targetWidth;
      }, 150);
    }
  });
});
