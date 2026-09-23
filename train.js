// ============================================================================
// EcoSort AI — Private Model Trainer & Exporter (train.js)
// Class order: Plastic = 0, Metal = 1, Neither = 2
// ============================================================================

const CLASSES = ["Plastic", "Metal", "Neither"];
const IMAGE_SIZE = 224; // MobileNetV2 expected input resolution
const BATCH_SIZE = 16;
const MAX_EPOCHS = 20;
const EARLY_STOP_PATIENCE = 3;
const TRAIN_FRACTION = 0.7;
const VAL_FRACTION = 0.15; // Remainder (0.15) becomes held-out test set
const SPLIT_SEED = 42;

const MOBILENET_URL =
  "https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_100_224/feature_vector/3/default/1";

const CLASS_FOLDERS = {
  Plastic: "dataset/plastic",
  Metal: "dataset/metal",
  Neither: "dataset/neither",
};

let featureExtractor = null; // Frozen MobileNetV2 GraphModel
let headModel = null; // Dense classification head

// ---------- DOM References ----------
const trainingStatusEl = document.getElementById("trainingStatus");
const progressFillEl = document.getElementById("progressFill");
const epochStatEl = document.getElementById("epochStat");
const lossStatEl = document.getElementById("lossStat");
const accStatEl = document.getElementById("accStat");
const valAccStatEl = document.getElementById("valAccStat");
const exportBtnEl = document.getElementById("exportBtn");

// ============================================================================
// 1. Memory-Safe Image Decoding via 2D Canvas
// ============================================================================
async function loadImageCanvas(path) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
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
    img.onerror = () => reject(new Error(`Could not load image: ${path}`));
    img.src = encodeURI(path);
  });
}

// ============================================================================
// 2. Training Augmentation (Memory-Safe, Train-Only)
// ============================================================================
function augmentImage(imageTensor) {
  return tf.tidy(() => {
    let img = imageTensor.expandDims(0); // [1, H, W, 3]

    if (Math.random() < 0.5) {
      img = tf.image.flipLeftRight(img);
    }

    const angle = (Math.random() - 0.5) * (Math.PI / 6); // ±15°
    img = tf.image.rotateWithOffset(img, angle, 0);

    const cropFrac = 0.85 + Math.random() * 0.15;
    const maxOffset = 1 - cropFrac;
    const y0 = Math.random() * maxOffset;
    const x0 = Math.random() * maxOffset;
    const boxes = tf.tensor2d([[y0, x0, y0 + cropFrac, x0 + cropFrac]]);
    const boxInd = tf.tensor1d([0], "int32");
    img = tf.image.cropAndResize(img, boxes, boxInd, [IMAGE_SIZE, IMAGE_SIZE]);

    let out = img.squeeze([0]);

    const brightnessDelta = (Math.random() - 0.5) * 0.15;
    out = out.add(brightnessDelta);

    const contrastFactor = 0.85 + Math.random() * 0.3;
    const mean = out.mean();
    out = out.sub(mean).mul(contrastFactor).add(mean);

    return out.clipByValue(0, 1);
  });
}

// ============================================================================
// 3. Manifest & Reproducible Dataset Splitting
// ============================================================================
async function loadManifest(folder) {
  const res = await fetch(`${folder}/manifest.json`);
  if (!res.ok) {
    throw new Error(`Could not load ${folder}/manifest.json (${res.status})`);
  }
  let filenames = await res.json();
  if (typeof filenames === "string") filenames = [filenames];
  return filenames.map((name) => `${folder}/${name}`);
}

function mulberry32(seed) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function seededShuffle(array, seed) {
  const rand = mulberry32(seed);
  const arr = array.slice();
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

async function buildExampleSplits() {
  const [plasticImages, metalImages, otherImages] = await Promise.all([
    loadManifest(CLASS_FOLDERS.Plastic),
    loadManifest(CLASS_FOLDERS.Metal),
    loadManifest(CLASS_FOLDERS.Neither),
  ]);

  let examples = [
    ...plasticImages.map((path) => ({ path, label: 0 })),
    ...metalImages.map((path) => ({ path, label: 1 })),
    ...otherImages.map((path) => ({ path, label: 2 })),
  ];

  // Deduplicate exact paths (normalized)
  const seen = new Set();
  examples = examples.filter((e) => {
    const normalized = e.path.toLowerCase().trim();
    if (seen.has(normalized)) return false;
    seen.add(normalized);
    return true;
  });

  examples = seededShuffle(examples, SPLIT_SEED);

  const total = examples.length;
  const trainCount = Math.round(total * TRAIN_FRACTION);
  const valCount = Math.round(total * VAL_FRACTION);

  return {
    trainExamples: examples.slice(0, trainCount),
    valExamples: examples.slice(trainCount, trainCount + valCount),
    testExamples: examples.slice(trainCount + valCount),
  };
}

function computeClassWeights(examples) {
  const counts = [0, 0, 0];
  examples.forEach((e) => counts[e.label]++);
  const total = examples.length;
  const weights = {};
  counts.forEach((c, i) => {
    weights[i] = c > 0 ? Math.min(total / (CLASSES.length * c), 5.0) : 1.0;
  });
  return weights;
}

// ============================================================================
// 4. Feature Precomputation
// ============================================================================
async function precomputeFeatureSet(
  examples,
  extractor,
  isTrain = false,
  onProgress = null,
) {
  const features = [];
  const labels = [];

  for (let i = 0; i < examples.length; i += BATCH_SIZE) {
    const chunk = examples.slice(i, i + BATCH_SIZE);
    const rawTensors = [];
    const chunkLabels = [];

    for (const { path, label } of chunk) {
      try {
        let t = await loadImageCanvas(path);
        if (isTrain) {
          const augmented = augmentImage(t);
          t.dispose();
          t = augmented;
        }
        rawTensors.push(t);
        chunkLabels.push(label);
      } catch (err) {
        console.warn("Skipping unreadable image:", path, err.message);
      }
      onProgress && onProgress();
    }

    if (rawTensors.length === 0) continue;

    const batchFeatures = tf.tidy(() => {
      const stacked = tf.stack(rawTensors);
      return extractor.predict(stacked);
    });

    const batchArray = await batchFeatures.array();

    batchFeatures.dispose();
    rawTensors.forEach((t) => t.dispose());

    batchArray.forEach((f, idx) => {
      features.push(f);
      labels.push(chunkLabels[idx]);
    });
  }

  return { features, labels };
}

function featureSetToDataset(featureSet, isTraining = false) {
  const items = featureSet.features.map((f, i) => ({
    f,
    label: featureSet.labels[i],
  }));

  let ds = tf.data.array(items);
  if (isTraining) {
    ds = ds.shuffle(items.length);
  }

  return ds.map((item) => ({
    xs: tf.tensor1d(item.f),
    ys: tf.oneHot(item.label, CLASSES.length),
  }));
}

// ============================================================================
// 5. Model Architecture & Regularization
// ============================================================================
function createHeadModel(inputDim) {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      inputShape: [inputDim],
      units: 128,
      activation: "relu",
      kernelRegularizer: tf.regularizers.l2({ l2: 1e-4 }),
    }),
  );
  model.add(tf.layers.dropout({ rate: 0.4 }));
  model.add(
    tf.layers.dense({
      units: CLASSES.length,
      activation: "softmax",
      kernelRegularizer: tf.regularizers.l2({ l2: 1e-4 }),
    }),
  );

  model.compile({
    optimizer: tf.train.adam(0.0005),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}

function argmax(probabilities) {
  let bestIndex = 0;
  for (let i = 1; i < probabilities.length; i++) {
    if (probabilities[i] > probabilities[bestIndex]) bestIndex = i;
  }
  return bestIndex;
}

function updateProgress(epoch, logs) {
  const pct = Math.round(((epoch + 1) / MAX_EPOCHS) * 100);
  progressFillEl.style.width = `${pct}%`;
  epochStatEl.textContent = `${epoch + 1} / ${MAX_EPOCHS}`;
  lossStatEl.textContent = logs.loss != null ? logs.loss.toFixed(3) : "—";

  const acc = logs.acc ?? logs.accuracy;
  const valAcc = logs.val_acc ?? logs.val_accuracy;

  accStatEl.textContent = acc != null ? `${(acc * 100).toFixed(1)}%` : "—";
  valAccStatEl.textContent =
    valAcc != null ? `${(valAcc * 100).toFixed(1)}%` : "—";

  console.log(`Epoch ${epoch + 1} tf.memory():`, tf.memory());
}

// ============================================================================
// 6. Test Set Evaluation
// ============================================================================
async function evaluateModel(testFeatureSet, headModel) {
  const matrix = Array.from({ length: CLASSES.length }, () =>
    new Array(CLASSES.length).fill(0),
  );

  for (let i = 0; i < testFeatureSet.features.length; i++) {
    const featTensor = tf.tensor2d([testFeatureSet.features[i]]);
    const pred = tf.tidy(() => headModel.predict(featTensor));
    const probs = await pred.data();

    featTensor.dispose();
    pred.dispose();

    const predicted = argmax(probs);
    const actual = testFeatureSet.labels[i];
    matrix[actual][predicted]++;
  }

  const total = testFeatureSet.labels.length;
  let correct = 0;
  for (let i = 0; i < CLASSES.length; i++) correct += matrix[i][i];
  const overallAcc = total > 0 ? correct / total : 0;

  const perClassAcc = CLASSES.map((_, i) => {
    const rowTotal = matrix[i].reduce((a, b) => a + b, 0);
    return rowTotal > 0 ? matrix[i][i] / rowTotal : null;
  });

  console.log("--- TEST SET EVALUATION ---");
  console.log("Confusion matrix (rows = actual, cols = predicted):");
  console.log("        " + CLASSES.map((c) => c.padEnd(8)).join(" "));
  matrix.forEach((row, i) =>
    console.log(
      CLASSES[i].padEnd(8),
      row.map((val) => String(val).padEnd(8)).join(" "),
    ),
  );

  return { overallAcc, perClassAcc, matrix };
}

// ============================================================================
// 7. Export Function
// ============================================================================
async function exportModel() {
  if (!headModel) return;
  try {
    trainingStatusEl.textContent = "Exporting trained model files…";
    // Triggers browser download of model.json and weight binary file
    await headModel.save("downloads://model");
    trainingStatusEl.textContent = "✅ Model exported! Move model.json and weight .bin files into /model/ directory.";
  } catch (err) {
    console.error("Export error:", err);
    trainingStatusEl.textContent = `⚠️ Export failed: ${err.message}`;
  }
}

exportBtnEl.addEventListener("click", exportModel);

// ============================================================================
// 8. Training Controller
// ============================================================================
async function train() {
  try {
    console.log("Initial tf.memory():", tf.memory());

    trainingStatusEl.textContent = "Loading MobileNetV2 feature extractor…";
    featureExtractor = await tf.loadGraphModel(MOBILENET_URL, {
      fromTFHub: true,
    });

    const dummy = tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3]);
    const dummyFeat = featureExtractor.predict(dummy);
    const featureDim = dummyFeat.shape[dummyFeat.shape.length - 1];
    dummy.dispose();
    dummyFeat.dispose();

    trainingStatusEl.textContent = "Reading dataset manifests…";
    const { trainExamples, valExamples, testExamples } =
      await buildExampleSplits();

    if (trainExamples.length === 0) {
      throw new Error("No training images found.");
    }

    const classWeight = computeClassWeights(trainExamples);

    let loadedSoFar = 0;
    const totalManifestImages =
      trainExamples.length + valExamples.length + testExamples.length;
    const reportProgress = () => {
      loadedSoFar++;
      trainingStatusEl.textContent = `Precomputing features… (${loadedSoFar} / ${totalManifestImages} images)`;
    };

    trainingStatusEl.textContent =
      "Precomputing training & validation features…";

    let trainFeatureSet = await precomputeFeatureSet(
      trainExamples,
      featureExtractor,
      true,
      reportProgress,
    );
    let valFeatureSet = await precomputeFeatureSet(
      valExamples,
      featureExtractor,
      false,
      reportProgress,
    );

    const trainDs = featureSetToDataset(trainFeatureSet, true).batch(
      BATCH_SIZE,
    );
    const valDs = featureSetToDataset(valFeatureSet, false).batch(BATCH_SIZE);

    headModel = createHeadModel(featureDim);
    headModel.summary();

    trainingStatusEl.textContent = "Training classifier head…";

    const customProgressCallback = new tf.CustomCallback({
      onEpochEnd: (epoch, logs) => updateProgress(epoch, logs),
    });

    await headModel.fitDataset(trainDs, {
      epochs: MAX_EPOCHS,
      validationData: valDs,
      classWeight,
      callbacks: [
        tf.callbacks.earlyStopping({
          monitor: "val_loss",
          patience: EARLY_STOP_PATIENCE,
        }),
        customProgressCallback,
      ],
    });

    trainingStatusEl.textContent = "Evaluating test set…";
    let testFeatureSet = await precomputeFeatureSet(
      testExamples,
      featureExtractor,
      false,
      reportProgress,
    );
    const { overallAcc } = await evaluateModel(testFeatureSet, headModel);

    // Free precomputed feature buffers
    trainFeatureSet = null;
    valFeatureSet = null;
    testFeatureSet = null;

    trainingStatusEl.textContent = `✅ Trained! Test accuracy: ${(
      overallAcc * 100
    ).toFixed(1)}%. Click below to export model files!`;
    exportBtnEl.disabled = false;

    console.log("Post-training tf.memory():", tf.memory());
  } catch (err) {
    console.error("Training error:", err);
    trainingStatusEl.textContent = `⚠️ ${err.message}`;
  }
}

train();
