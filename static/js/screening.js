/**
 * DeepGuard X — Screening and Webcam Interactive Logic
 */

let videoStream = null;

function initWebcam(videoElementId, canvasElementId, captureBtnId, retakeBtnId, analyzeBtnId, hiddenInputId) {
  const video = document.getElementById(videoElementId);
  const canvas = document.getElementById(canvasElementId);
  const captureBtn = document.getElementById(captureBtnId);
  const retakeBtn = document.getElementById(retakeBtnId);
  const analyzeBtn = document.getElementById(analyzeBtnId);
  const hiddenInput = document.getElementById(hiddenInputId);
  const statusBadge = document.getElementById("webcam-status-badge");

  if (!video || !canvas) return;

  // Start Camera
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: "user" } })
      .then(function (stream) {
        videoStream = stream;
        video.srcObject = stream;
        video.play();
        if (statusBadge) {
          statusBadge.textContent = "CAMERA LIVE";
          statusBadge.className = "inline-flex items-center rounded-full bg-emerald-500/10 px-3 py-1 text-xs font-semibold text-emerald-400 border border-emerald-500/30";
        }
      })
      .catch(function (err) {
        console.error("Camera access error:", err);
        if (statusBadge) {
          statusBadge.textContent = "CAMERA UNAVAILABLE";
          statusBadge.className = "inline-flex items-center rounded-full bg-rose-500/10 px-3 py-1 text-xs font-semibold text-rose-400 border border-rose-500/30";
        }
      });
  }

  // Capture Button
  if (captureBtn) {
    captureBtn.addEventListener("click", function () {
      const context = canvas.getContext("2d");
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      context.drawImage(video, 0, 0, canvas.width, canvas.height);

      const dataUrl = canvas.toDataURL("image/jpeg", 0.95);
      if (hiddenInput) {
        hiddenInput.value = dataUrl;
      }

      video.classList.add("hidden");
      canvas.classList.remove("hidden");
      captureBtn.classList.add("hidden");
      if (retakeBtn) retakeBtn.classList.remove("hidden");
      if (analyzeBtn) analyzeBtn.classList.remove("hidden");
    });
  }

  // Retake Button
  if (retakeBtn) {
    retakeBtn.addEventListener("click", function () {
      canvas.classList.add("hidden");
      video.classList.remove("hidden");
      if (captureBtn) captureBtn.classList.remove("hidden");
      retakeBtn.classList.add("hidden");
      if (analyzeBtn) analyzeBtn.classList.add("hidden");
      if (hiddenInput) hiddenInput.value = "";
    });
  }
}
