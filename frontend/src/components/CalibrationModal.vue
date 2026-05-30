<template>
  <div class="modal-overlay">
    <div class="modal">
      <div ref="cameraBoxRef" class="modal-media">
        <video ref="videoRef" autoplay playsinline class="camera" />
        <div class="bbox-layer">
          <div
            v-for="(box, index) in boxes"
            :key="index"
            class="bbox"
            :style="getBoxStyle(box)"
          ></div>
        </div>
      </div>

      <div class="modal-panel">
        <p class="eyebrow">Calibration</p>
        <h2 class="title">
          {{ currentStepConfig.displayLabel }}
        </h2>

        <h3 v-if="!completed && !isUploading" class="instruction">
          {{ currentStepConfig.instruction }}
        </h3>

        <h1 v-if="countdown > 0 && !isCapturing && !isUploading && !completed" class="countdown">
          {{ countdown }}
        </h1>

        <p v-if="isCapturing" class="status">
          Capturing {{ currentStepConfig.displayLabel }}: {{ captureCount }}/100
        </p>

        <p v-if="isUploading" class="status">Running calibration...</p>
        <p v-if="statusMessage" class="status subtle">{{ statusMessage }}</p>

        <p v-if="metrics" class="metrics">
          Base: {{ formatAccuracy(metrics.base_accuracy) }} | Ridge:
          {{ formatAccuracy(metrics.ridge_accuracy) }}
        </p>

        <button
          v-if="!isCapturing && !isCountingDown && !isUploading && !completed"
          class="capture-btn"
          @click="startCountdown"
        >
          Start Capture
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from "vue";
import { onMounted, onUnmounted } from "vue";
import { useEyeStore } from "@/store/eyeStore";

const emit = defineEmits(["close", "complete"]);
const { ensureSessionId, setSessionId } = useEyeStore();

const steps = [
  { wireLabel: "left", displayLabel: "LEFT", instruction: "Look LEFT" },
  { wireLabel: "right", displayLabel: "RIGHT", instruction: "Look RIGHT" },
  { wireLabel: "up", displayLabel: "UP", instruction: "Look UP" },
  { wireLabel: "down", displayLabel: "DOWN", instruction: "Look DOWN" },
  { wireLabel: "straight", displayLabel: "OPEN", instruction: "Open your eyes" },
  { wireLabel: "closed", displayLabel: "CLOSED", instruction: "Close your eyes" },
];

const currentStep = ref(0);
const countdown = ref(3);
const isCountingDown = ref(false);
const isCapturing = ref(false);
const isUploading = ref(false);
const completed = ref(false);
const captureCount = ref(0);
const statusMessage = ref("");
const metrics = ref(null);
const videoRef = ref(null);
const cameraBoxRef = ref(null);
const boxes = ref([]);
const frameSize = ref({ width: 1, height: 1 });

let stream = null;
const sessionId = ensureSessionId();

// Play a sound when a calibration step is completed to prompt the user to move to the next position
const playBeep = () => {
  const audio = new Audio("/sounds/ding.mp3"); // or any built-in sound
  audio.play().catch(() => {
    err => console.log("Audio blocked:", err)
  });
};

const currentStepConfig = computed(() => steps[currentStep.value] ?? steps[steps.length - 1]);

const startCamera = async () => {
  stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { min: 640, ideal: 1920, max: 1920 },
      height: { min: 480, ideal: 1080, max: 1080 },
      frameRate: { ideal: 30, max: 30 },
    },
  });

  videoRef.value.srcObject = stream;
  await videoRef.value.play();
};

const stopCamera = () => {
  if (!stream) {
    return;
  }
  stream.getTracks().forEach((track) => track.stop());
  stream = null;
};

onMounted(() => {
  void startCamera();
});

onUnmounted(() => {
  stopCamera();
});

const startCountdown = () => {
  isCountingDown.value = true;
  countdown.value = 3;
  statusMessage.value = "";

  const interval = setInterval(() => {
    countdown.value -= 1;
    if (countdown.value > 0) {
      return;
    }

    clearInterval(interval);
    isCountingDown.value = false;
    void captureCurrentLabel();
  }, 1000);
};

const captureVideoFrame = async () => {
  const video = videoRef.value;
  if (!video || !video.videoWidth || !video.videoHeight) {
    throw new Error("Camera is not ready");
  }

  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0);

  const blob = await new Promise((resolve) => {
    canvas.toBlob(resolve, "image/jpeg", 1.0);
  });

  if (!blob) {
    throw new Error("Failed to capture calibration frame");
  }

  return blob.arrayBuffer();
};

const captureCurrentLabel = async () => {
  const { wireLabel, displayLabel } = currentStepConfig.value;
  const reset = currentStep.value === 0;

  isCapturing.value = true;
  captureCount.value = 0;
  statusMessage.value = `Connecting for ${displayLabel}...`;

  const wsBase = import.meta.env.VITE_API_BASE_URL.replace(/^http/, "ws");
  const socket = new WebSocket(`${wsBase}/ws/calibration`);
  socket.binaryType = "arraybuffer";

  try {
    await new Promise((resolve, reject) => {
      socket.onopen = resolve;
      socket.onerror = () => reject(new Error("Calibration socket failed to open"));
    });

    socket.send(JSON.stringify({ session_id: sessionId, label: wireLabel, reset }));

    await new Promise((resolve, reject) => {
      socket.onmessage = async (event) => {
        const data = JSON.parse(event.data);

        if (data.status === "error") {
          reject(new Error(data.detail || "Calibration failed"));
          return;
        }

        if (typeof data.captured_count === "number") {
          captureCount.value = data.captured_count;
        }
        boxes.value = Array.isArray(data.boxes) ? data.boxes : [];
        if (data.frame_size?.width && data.frame_size?.height) {
          frameSize.value = data.frame_size;
        }

        if (data.status === "ready" || data.status === "capturing" || data.status === "retry") {
          statusMessage.value =
            data.status === "retry"
              ? data.reason || "No eye detected, retrying..."
              : `Capturing ${displayLabel}...`;

          try {
            socket.send(await captureVideoFrame());
          } catch (error) {
            reject(error);
          }
          return;
        }

        if (data.status === "complete") {
          resolve();
        }
      };

      socket.onclose = () => {
        if (captureCount.value < 100) {
          reject(new Error("Calibration socket closed before completion"));
        }
      };

      socket.onerror = () => reject(new Error("Calibration socket error"));
    });

    socket.close();
    isCapturing.value = false;
    boxes.value = [];
    statusMessage.value = `${displayLabel} complete`;

    currentStep.value += 1;
    if (currentStep.value < steps.length) {
      playBeep();
      return;
    }

    await finishCalibration();
  } catch (error) {
    socket.close();
    isCapturing.value = false;
    statusMessage.value = error instanceof Error ? error.message : "Calibration failed";
  }
};

const finishCalibration = async () => {
  isUploading.value = true;
  statusMessage.value = "Training personalized model...";

  try {
    const response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/sessions/${sessionId}/calibration`, {
      method: "POST",
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Calibration failed");
    }

    metrics.value = data;
    completed.value = true;
    setSessionId(sessionId);
    statusMessage.value = "Calibration successful";
    emit("complete", sessionId);

    setTimeout(() => {
      stopCamera();
      emit("close");
    }, 3000);
  } catch (error) {
    statusMessage.value = error instanceof Error ? error.message : "Calibration failed";
  } finally {
    isUploading.value = false;
  }
};

const formatAccuracy = (value) => `${((value ?? 0) * 100).toFixed(1)}%`;

const getBoxStyle = (box) => {
  const container = cameraBoxRef.value;
  if (!container || !box) {
    return {};
  }

  const containerWidth = container.clientWidth;
  const containerHeight = container.clientHeight;
  const sourceWidth = frameSize.value.width;
  const sourceHeight = frameSize.value.height;
  const scale = Math.max(containerWidth / sourceWidth, containerHeight / sourceHeight);
  const renderedWidth = sourceWidth * scale;
  const renderedHeight = sourceHeight * scale;
  const offsetX = (containerWidth - renderedWidth) / 2;
  const offsetY = (containerHeight - renderedHeight) / 2;
  return {
    left: `${offsetX + box.x1 * scale}px`,
    top: `${offsetY + box.y1 * scale}px`,
    width: `${(box.x2 - box.x1) * scale}px`,
    height: `${(box.y2 - box.y1) * scale}px`,
  };
};
</script>

<style scoped>
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  justify-content: center;
  align-items: center;
}

.modal {
  background: #1e293b;
  padding: 24px;
  border-radius: 18px;
  width: min(980px, calc(100vw - 48px));
  display: grid;
  grid-template-columns: minmax(0, 1.6fr) minmax(280px, 0.9fr);
  gap: 24px;
  align-items: center;
  box-shadow: 0 24px 60px rgba(0, 0, 0, 0.35);
}

.modal-media {
  position: relative;
  background: #0f172a;
  border-radius: 16px;
  overflow: hidden;
  min-height: 320px;
}

.camera {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.bbox-layer {
  position: absolute;
  inset: 0;
  pointer-events: none;
}

.bbox {
  position: absolute;
  border: 2px solid #22c55e;
  border-radius: 8px;
  box-shadow: 0 0 0 1px rgba(15, 23, 42, 0.5);
}

.modal-panel {
  color: white;
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 12px;
}

.eyebrow {
  margin: 0;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  font-size: 12px;
  color: #93c5fd;
}

.title {
  margin: 0;
  font-size: 34px;
  line-height: 1;
}

.instruction {
  margin: 0;
  font-size: 22px;
  color: #e2e8f0;
}

.countdown {
  margin: 0;
  font-size: 76px;
  line-height: 1;
  color: #22c55e;
}

.status {
  margin: 0;
  font-size: 16px;
  color: #f8fafc;
}

.subtle {
  color: #cbd5e1;
}

.metrics {
  margin: 0;
  color: #f8fafc;
  font-weight: 600;
}

.capture-btn {
  margin-top: 8px;
  border: none;
  border-radius: 12px;
  padding: 14px 18px;
  background: #3b82f6;
  color: white;
  font-weight: 700;
  font-size: 15px;
  cursor: pointer;
  transition: transform 0.15s ease, opacity 0.15s ease;
}

.capture-btn:hover {
  transform: translateY(-1px);
}

.capture-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

@media (max-width: 860px) {
  .modal {
    grid-template-columns: 1fr;
  }

  .modal-media {
    min-height: 220px;
  }
}
</style>
