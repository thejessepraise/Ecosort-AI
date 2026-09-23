// ============================================================================
// EcoSort AI — Public Inference Engine (script.js)
// Class order: Plastic = 0, Metal = 1, Neither = 2
// ============================================================================

const CLASSES = ["Plastic", "Metal", "Neither"];
const IMAGE_SIZE = 224; // MobileNetV2 expected input resolution

// --- Configurable Uncertainty Thresholds ---
const CONFIDENCE_THRESHOLD = 0.55;
const MARGIN_THRESHOLD = 0.20;

const MOBILENET_URL =
  "https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_100_224/feature_vector/3/default/1";

const MODEL_JSON_PATH = "./model/model.json";

const TIPS = {
  Plastic: "♻️ Rinse it out and pop it in your plastics/recycling bin.",
  Metal: "🥫 Metal is infinitely recyclable — drop it in the metal recycling bin.",
  Neither: "🤷 This doesn't look like plastic or metal waste — no recycling tip here!",
};

let featureExtractor = null; // Frozen MobileNetV2 GraphModel
let headModel = null; // Trained Dense classification head
let isClassifying = false;

// ---------- DOM References ----------
const trainingStatusEl = document.getElementById("trainingStatus");
const dropzoneEl = document.getElementById("dropzone");
const dropzoneHintEl = document.getElementById("dropzoneHint");
const fileInputEl = document.getElementById("fileInput");
const previewEl = document.getElementById("preview");
const classifyBtnEl = document.getElementById("classifyBtn");
const resultBoxEl = document.getElementById("resultBox");
const resultLabelEl = document.getElementById("resultLabel");
const confidenceFillEl = document.getElementById("confidenceFill");
const resultConfidenceEl =
  document.getElementById("resultConfidenceEl") ||
  document.getElementById("resultConfidence");
const resultTipEl = document.getElementById("resultTip");

let selectedFile = null;

// ============================================================================
// 1. Memory-Safe Image Decoding via 2D Canvas
// ============================================================================
async function loadImageFromFileCanvas(file) {
  const url = URL.createObjectURL(file);
  try {
    return await new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement("canvas");
        canvas.width = IMAGE_SIZE;
        canvas.height = IMAGE_SIZE;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0, IMAGE_SIZE, IMAGE_SIZE);

        const tensor = tf.tidy(() => {
          return tf.browser.fromPixels(canvas, 3).toFloat().div(255);
        });
        resolve(tensor);
      };
      img.onerror = () => reject(new Error("Could not read image file."));
      img.src = url;
    });
  } finally {
    URL.revokeObjectURL(url);
  }
}

function argmax(probabilities) {
  let bestIndex = 0;
  for (let i = 1; i < probabilities.length; i++) {
    if (probabilities[i] > probabilities[bestIndex]) bestIndex = i;
  }
  return bestIndex;
}

// ============================================================================
// 2. Model Initialization
// ============================================================================
async function initInferenceEngine() {
  try {
    trainingStatusEl.textContent = "Loading pretrained MobileNetV2 feature extractor…";
    featureExtractor = await tf.loadGraphModel(MOBILENET_URL, {
      fromTFHub: true,
    });

    trainingStatusEl.textContent = "Loading trained EcoSort classification head…";
    try {
      headModel = await tf.loadLayersModel(MODEL_JSON_PATH);
      trainingStatusEl.textContent = "✅ AI Model Ready! Upload a photo to classify waste.";
      classifyBtnEl.disabled = !selectedFile;
    } catch (loadErr) {
      console.warn("Could not load model from", MODEL_JSON_PATH, loadErr.message);
      trainingStatusEl.textContent =
        "⚠️ Pretrained model file (model/model.json) not found. Run train.html locally first and export model files to /model/.";
    }
  } catch (err) {
    console.error("Initialization error:", err);
    trainingStatusEl.textContent = `⚠️ Initialization error: ${err.message}`;
  }
}

// ============================================================================
// 3. UI Event Handlers & Inference
// ============================================================================
function setPreview(file) {
  const url = URL.createObjectURL(file);
  previewEl.src = url;
  previewEl.hidden = false;
  dropzoneHintEl.hidden = true;
}

function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  selectedFile = file;
  setPreview(file);
  classifyBtnEl.disabled = !(headModel && featureExtractor);
  resultBoxEl.hidden = true;
}

dropzoneEl.addEventListener("click", () => fileInputEl.click());

fileInputEl.addEventListener("change", (e) => {
  handleFile(e.target.files[0]);
});

["dragover", "dragenter"].forEach((evt) =>
  dropzoneEl.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzoneEl.classList.add("dragover");
  }),
);

["dragleave", "drop"].forEach((evt) =>
  dropzoneEl.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzoneEl.classList.remove("dragover");
  }),
);

dropzoneEl.addEventListener("drop", (e) => {
  handleFile(e.dataTransfer.files[0]);
});

classifyBtnEl.addEventListener("click", async () => {
  if (isClassifying || !headModel || !featureExtractor || !selectedFile) return;
  isClassifying = true;
  classifyBtnEl.disabled = true;
  classifyBtnEl.textContent = "Classifying…";

  let rawTensor = null;
  let featTensor = null;
  let predictionTensor = null;

  try {
    // 1. Decode via 2D Canvas (224x224, normalized to [0,1])
    rawTensor = await loadImageFromFileCanvas(selectedFile);

    // 2. Extract features via MobileNetV2
    featTensor = tf.tidy(() =>
      featureExtractor.predict(rawTensor.expandDims(0)),
    );

    // 3. Predict class probabilities via loaded classification head
    predictionTensor = tf.tidy(() => headModel.predict(featTensor));
    const probabilities = await predictionTensor.data();

    const predictedIndex = argmax(probabilities);
    const label = CLASSES[predictedIndex];

    // Margin analysis (top prob - 2nd top prob)
    const sortedProbs = [...probabilities].sort((a, b) => b - a);
    const topProb = sortedProbs[0];
    const secondProb = sortedProbs[1];
    const margin = topProb - secondProb;

    resultBoxEl.classList.remove("result-uncertain");
    confidenceFillEl.style.width = `${(topProb * 100).toFixed(1)}%`;

    // 4. Robust Uncertainty Check
    if (topProb < CONFIDENCE_THRESHOLD || margin < MARGIN_THRESHOLD) {
      resultBoxEl.classList.add("result-uncertain");
      resultLabelEl.textContent = "🤔 Not sure";
      resultConfidenceEl.textContent = `Prediction strength ${(
        topProb * 100
      ).toFixed(
        1,
      )}% (margin ${(margin * 100).toFixed(1)}%) — insufficient certainty between categories.`;
      resultTipEl.textContent =
        "Try a clearer, well-lit photo of the single item.";
    } else if (label === "Neither") {
      resultLabelEl.textContent = "🤷 Not plastic or metal";
      resultConfidenceEl.textContent = `Prediction strength: ${(topProb * 100).toFixed(1)}%.`;
      resultTipEl.textContent = TIPS.Neither;
    } else {
      resultLabelEl.textContent = label;
      resultConfidenceEl.textContent = `Prediction strength: ${(topProb * 100).toFixed(1)}%.`;
      resultTipEl.textContent = TIPS[label] ?? "";
    }

    resultBoxEl.hidden = false;
    console.log("Post-classification tf.memory():", tf.memory());
  } catch (err) {
    console.error("Classification error:", err);
    resultLabelEl.textContent = "Error";
    resultConfidenceEl.textContent = err.message;
    resultTipEl.textContent = "";
    resultBoxEl.hidden = false;
  } finally {
    if (rawTensor) rawTensor.dispose();
    if (featTensor) featTensor.dispose();
    if (predictionTensor) predictionTensor.dispose();

    isClassifying = false;
    classifyBtnEl.disabled = false;
    classifyBtnEl.textContent = "Classify Waste";
  }
});

initInferenceEngine();
