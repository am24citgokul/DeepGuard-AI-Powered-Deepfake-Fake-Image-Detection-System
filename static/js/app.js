/**
 * DeepGuard X — Enterprise AI Authenticity Screening Frontend Scripts
 */

document.addEventListener("DOMContentLoaded", function () {
  initFileUploadZones();
  initFormSubmissions();
});

/**
 * Enhanced Drag & Drop and File Preview Handlers
 */
function initFileUploadZones() {
  const dropzones = document.querySelectorAll(".dg-dropzone");

  dropzones.forEach(function (zone) {
    const input = zone.querySelector("input[type='file']");
    const previewContainer = zone.querySelector(".file-preview-area");
    const defaultPlaceholder = zone.querySelector(".upload-placeholder");
    const removeBtn = zone.querySelector(".btn-remove-file");

    if (!input) return;

    // Highlight drop area when item is dragged over it
    ["dragenter", "dragover"].forEach(function (eventName) {
      zone.addEventListener(eventName, function (e) {
        e.preventDefault();
        e.stopPropagation();
        zone.classList.add("drag-active");
      });
    });

    ["dragleave", "drop"].forEach(function (eventName) {
      zone.addEventListener(eventName, function (e) {
        e.preventDefault();
        e.stopPropagation();
        zone.classList.remove("drag-active");
      });
    });

    // Handle dropped files
    zone.addEventListener("drop", function (e) {
      const dt = e.dataTransfer;
      const files = dt.files;
      if (files.length > 0) {
        input.files = files;
        handleFileSelection(input, zone);
      }
    });

    // Handle file input change
    input.addEventListener("change", function () {
      handleFileSelection(input, zone);
    });

    // Remove file button
    if (removeBtn) {
      removeBtn.addEventListener("click", function (e) {
        e.preventDefault();
        e.stopPropagation();
        input.value = "";
        if (previewContainer) previewContainer.classList.add("hidden");
        if (defaultPlaceholder) defaultPlaceholder.classList.remove("hidden");
      });
    }
  });
}

function handleFileSelection(input, zone) {
  const previewContainer = zone.querySelector(".file-preview-area");
  const defaultPlaceholder = zone.querySelector(".upload-placeholder");
  const filenameEl = zone.querySelector(".file-name-display");
  const filesizeEl = zone.querySelector(".file-size-display");
  const imagePreview = zone.querySelector(".img-preview-thumb");

  if (input.files && input.files[0]) {
    const file = input.files[0];
    if (filenameEl) filenameEl.textContent = file.name;
    if (filesizeEl) filesizeEl.textContent = formatBytes(file.size);

    if (imagePreview && file.type.startsWith("image/")) {
      const reader = new FileReader();
      reader.onload = function (e) {
        imagePreview.src = e.target.result;
        imagePreview.classList.remove("hidden");
      };
      reader.readAsDataURL(file);
    } else if (imagePreview) {
      imagePreview.classList.add("hidden");
    }

    if (defaultPlaceholder) defaultPlaceholder.classList.add("hidden");
    if (previewContainer) previewContainer.classList.remove("hidden");
  }
}

function formatBytes(bytes, decimals = 1) {
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + " " + sizes[i];
}

/**
 * Handle Loading Overlay during Model Inference
 */
function initFormSubmissions() {
  const forms = document.querySelectorAll("form.screen-analysis-form");
  const loadingOverlay = document.getElementById("analysis-loading-overlay");

  forms.forEach(function (form) {
    form.addEventListener("submit", function () {
      if (loadingOverlay) {
        loadingOverlay.classList.remove("hidden");
        loadingOverlay.classList.add("flex");
      }
    });
  });
}
