const IMAGE_SIZE = 64;
const CLASSES = ["Plastic", "Metal"];
const EPOCHS = 30;
const VALIDATION_SPLIT = 0.2;

// INTEGRITY GUARDRAIL: this model only ever has two options to choose from,
// so softmax forces its outputs to sum to 100% even for a photo of neither
// material — it has no built-in way to say "I don't know." As a practical
// safety net (not a perfect fix — see the caveat in chat), we refuse to
// commit to a label when the model's own top confidence is weak, since a
// near-50/50 split is the model itself signaling it's torn.
const CONFIDENCE_THRESHOLD = 0.9;

const TIPS = {
  Plastic: "♻️ Rinse it out and pop it in your plastics/recycling bin.",
  Metal:
    "🥫 Metal is infinitely recyclable — drop it in the metal recycling bin.",
};

const plasticImages = [
  "dataset/plastic/3c171372e2ef27248f5f99e66231e1d9.jpg",
  "dataset/plastic/images.jpg",
  "dataset/plastic/images (1).jpg",
  "dataset/plastic/images (2).jpg",
  "dataset/plastic/images (5).jpg",
  "dataset/plastic/images (7).jpg",
  "dataset/plastic/images (8).jpg",
  "dataset/plastic/images (9).jpg",
  "dataset/plastic/images (10).jpg",
];

const metalImages = [
  "dataset/metal/images.jpg",
  "dataset/metal/images (1).jpg",
  "dataset/metal/images (2).jpg",
  "dataset/metal/images (3).jpg",
  "dataset/metal/images (4).jpg",
  "dataset/metal/images (7).jpg",
];

let model = null;

// ---------- DOM references ----------
const trainingStatusEl = document.getElementById("trainingStatus");
const progressFillEl = document.getElementById("progressFill");
const epochStatEl = document.getElementById("epochStat");
const lossStatEl = document.getElementById("lossStat");
const accStatEl = document.getElementById("accStat");
const valAccStatEl = document.getElementById("valAccStat");

const dropzoneEl = document.getElementById("dropzone");
const dropzoneHintEl = document.getElementById("dropzoneHint");
const fileInputEl = document.getElementById("fileInput");
const previewEl = document.getElementById("preview");
const classifyBtnEl = document.getElementById("classifyBtn");
const resultBoxEl = document.getElementById("resultBox");
const resultLabelEl = document.getElementById("resultLabel");
const confidenceFillEl = document.getElementById("confidenceFill");
const resultConfidenceEl = document.getElementById("resultConfidence");
const resultTipEl = document.getElementById("resultTip");

let selectedFile = null;

// Load an image from a URL and convert it into a normalized Tensor.
// BUG FIX: the original code never handled a failed image load, so a
// missing/blocked file would silently hang the whole training pipeline
// forever. We now reject with a clear error instead.
async function loadImage(path) {
  const img = new Image();
  img.src = encodeURI(path);

  await new Promise((resolve, reject) => {
    img.onload = resolve;
    img.onerror = () => reject(new Error(`Could not load image: ${path}`));
  });

  return tf.tidy(() => {
    let tensor = tf.browser.fromPixels(img, 3);
    tensor = tf.image.resizeBilinear(tensor, [IMAGE_SIZE, IMAGE_SIZE]);
    tensor = tensor.div(255);
    return tensor;
  });
}

// Load all training images.
async function createDataset() {
  const inputs = [];
  const outputs = [];

  for (const path of plasticImages) {
    inputs.push(await loadImage(path)); // Plastic = 0
    outputs.push(0);
  }

  for (const path of metalImages) {
    inputs.push(await loadImage(path)); // Metal = 1
    outputs.push(1);
  }

  return { inputs, outputs };
}

// BUG FIX: the dataset is built class-by-class (all plastic, then all
// metal). tf.js's `validationSplit` always carves its validation set off
// the *end* of the data it's given, so without shuffling first, the
// "validation set" would have been 100% metal images and accuracy numbers
// would have been meaningless. We shuffle inputs/outputs together
// (Fisher-Yates, keeping each image paired with its correct label) before
// handing anything to the model.
function shuffleDataset(dataset) {
  const { inputs, outputs } = dataset;
  for (let i = inputs.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [inputs[i], inputs[j]] = [inputs[j], inputs[i]];
    [outputs[i], outputs[j]] = [outputs[j], outputs[i]];
  }
  return dataset;
}

// Build the CNN.
function createModel() {
  const model = tf.sequential();

  model.add(
    tf.layers.conv2d({
      inputShape: [IMAGE_SIZE, IMAGE_SIZE, 3],
      filters: 16,
      kernelSize: 3,
      padding: "same",
      activation: "relu",
    }),
  );
  model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

  model.add(
    tf.layers.conv2d({
      filters: 32,
      kernelSize: 3,
      padding: "same",
      activation: "relu",
    }),
  );
  model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

  model.add(tf.layers.flatten());

  // ADDED: dropout. With only a handful of training images, the original
  // model had nothing to stop it from just memorizing them (overfitting).
  // Dropping 30% of the dense-layer activations each step forces it to
  // learn sturdier, more general features instead.
  model.add(tf.layers.dense({ units: 64, activation: "relu" }));
  model.add(tf.layers.dropout({ rate: 0.3 }));

  model.add(tf.layers.dense({ units: CLASSES.length, activation: "softmax" }));

  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}

function updateProgress(epoch, logs) {
  const pct = Math.round(((epoch + 1) / EPOCHS) * 100);
  progressFillEl.style.width = `${pct}%`;
  epochStatEl.textContent = `${epoch + 1} / ${EPOCHS}`;
  lossStatEl.textContent = logs.loss.toFixed(3);

  const acc = logs.acc ?? logs.accuracy;
  const valAcc = logs.val_acc ?? logs.val_accuracy;

  accStatEl.textContent = acc != null ? `${(acc * 100).toFixed(1)}%` : "—";
  valAccStatEl.textContent =
    valAcc != null ? `${(valAcc * 100).toFixed(1)}%` : "—";
}

async function train() {
  try {
    trainingStatusEl.textContent = "Loading dataset…";
    const dataset = await createDataset();

    trainingStatusEl.textContent = `Loaded ${dataset.inputs.length} images. Shuffling…`;
    shuffleDataset(dataset);

    const inputs = tf.stack(dataset.inputs);
    const outputs = tf.oneHot(
      tf.tensor1d(dataset.outputs, "int32"),
      CLASSES.length,
    );

    model = createModel();
    model.summary();

    trainingStatusEl.textContent = "Training…";

    await model.fit(inputs, outputs, {
      epochs: EPOCHS,
      // BUG FIX: `shuffle: true` here only reshuffles the *training*
      // portion on every epoch; it does not fix which images become the
      // validation set (that's decided once, from the tail of the array,
      // before this shuffling happens) — hence shuffling the raw dataset
      // ourselves above.
      shuffle: true,
      validationSplit: VALIDATION_SPLIT,
      batchSize: 4,
      callbacks: {
        onEpochEnd: (epoch, logs) => updateProgress(epoch, logs),
      },
    });

    trainingStatusEl.textContent =
      "✅ Model trained! Try classifying your own photo below.";
    classifyBtnEl.disabled = !selectedFile;

    // Demonstrate on a genuine held-out example, i.e. one of the images
    // from the validation slice tf.js itself never trained on.
    // BUG FIX: the original code evaluated on a random *training* image,
    // which is data leakage — the model had already seen the answer, so
    // that "test" couldn't tell you anything about real accuracy.
    const valCount = Math.max(
      1,
      Math.round(dataset.inputs.length * VALIDATION_SPLIT),
    );
    const holdoutStart = dataset.inputs.length - valCount;
    const holdoutIndex = holdoutStart + Math.floor(Math.random() * valCount);

    await testModel(model, dataset, holdoutIndex);

    inputs.dispose();
    outputs.dispose();
    dataset.inputs.forEach((tensor) => tensor.dispose());
  } catch (err) {
    console.error(err);
    trainingStatusEl.textContent = `⚠️ ${err.message}`;
  }
}

async function testModel(model, dataset, index) {
  const image = dataset.inputs[index];

  const prediction = tf.tidy(() => model.predict(image.expandDims(0)));
  const probabilities = await prediction.data();
  prediction.dispose();

  const predictedIndex = probabilities[0] > probabilities[1] ? 0 : 1;
  const confidence = probabilities[predictedIndex] * 100;
  const actual = CLASSES[dataset.outputs[index]];
  const correct = predictedIndex === dataset.outputs[index];

  console.log("Held-out sample — actual:", actual);
  console.log("Held-out sample — prediction:", CLASSES[predictedIndex]);
  console.log("Probabilities:", probabilities);

  trainingStatusEl.textContent =
    `✅ Trained! Held-out check → predicted ${CLASSES[predictedIndex]} ` +
    `(${confidence.toFixed(1)}%), actual ${actual} ${correct ? "✔️" : "✖️"}`;
}

// ---------- Upload & classify UI ----------

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
  classifyBtnEl.disabled = !model;
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

async function loadImageFromFile(file) {
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.src = url;

  await new Promise((resolve, reject) => {
    img.onload = resolve;
    img.onerror = () => reject(new Error("Could not read that image file."));
  });

  const tensor = tf.tidy(() => {
    let t = tf.browser.fromPixels(img, 3);
    t = tf.image.resizeBilinear(t, [IMAGE_SIZE, IMAGE_SIZE]);
    return t.div(255);
  });

  URL.revokeObjectURL(url);
  return tensor;
}

classifyBtnEl.addEventListener("click", async () => {
  if (!model || !selectedFile) return;

  classifyBtnEl.disabled = true;
  classifyBtnEl.textContent = "Classifying…";

  try {
    const tensor = await loadImageFromFile(selectedFile);

    const prediction = tf.tidy(() => model.predict(tensor.expandDims(0)));
    const probabilities = await prediction.data();
    tensor.dispose();
    prediction.dispose();

    const predictedIndex = probabilities[0] > probabilities[1] ? 0 : 1;
    const confidence = probabilities[predictedIndex];

    resultBoxEl.classList.remove("result-uncertain");
    confidenceFillEl.style.width = `${(confidence * 100).toFixed(1)}%`;

    if (confidence < CONFIDENCE_THRESHOLD) {
      // The model itself is torn between the two options — treat that as
      // "this probably isn't confidently plastic or metal" rather than
      // forcing a guess.
      resultBoxEl.classList.add("result-uncertain");
      resultLabelEl.textContent = "🤔 Not sure";
      resultConfidenceEl.textContent = `Only ${(confidence * 100).toFixed(1)}% confident — this may not be plastic or metal at all.`;
      resultTipEl.textContent =
        "This model only knows plastic vs. metal, so anything else will confuse it. Try a clearer photo, or one of just plastic/metal waste.";
    } else {
      const label = CLASSES[predictedIndex];
      resultLabelEl.textContent = label;
      resultConfidenceEl.textContent = `${(confidence * 100).toFixed(1)}% confident`;
      resultTipEl.textContent = TIPS[label] ?? "";
    }

    resultBoxEl.hidden = false;
  } catch (err) {
    console.error(err);
    resultLabelEl.textContent = "Error";
    resultConfidenceEl.textContent = err.message;
    resultTipEl.textContent = "";
    resultBoxEl.hidden = false;
  } finally {
    classifyBtnEl.disabled = false;
    classifyBtnEl.textContent = "Classify Waste";
  }
});

train();
