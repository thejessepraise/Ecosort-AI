const IMAGE_SIZE = 64;

const CLASSES = ["Plastic", "Metal"];

const plasticImages = [
  "/dataset/plastic/3c171372e2ef27248f5f99e66231e1d9.jpg",
  "/dataset/plastic/images.jpg",
  "/dataset/plastic/images (1).jpg",
  "/dataset/plastic/images (2).jpg",
  "/dataset/plastic/images (5).jpg",
  "/dataset/plastic/images (7).jpg",
  "/dataset/plastic/images (8).jpg",
  "/dataset/plastic/images (9).jpg",
  "/dataset/plastic/images (10).jpg",
];

const metalImages = [
  "/dataset/metal/images.jpg",
  "/dataset/metal/images (1).jpg",
  "/dataset/metal/images (2).jpg",
  "/dataset/metal/images (3).jpg",
  "/dataset/metal/images (4).jpg",
  "/dataset/metal/images (7).jpg",
];

// Load an image and convert it into a Tensor
async function loadImage(path) {
  const img = new Image();

  img.src = path;

  await new Promise((resolve) => {
    img.onload = resolve;
  });

  return tf.tidy(() => {
    // Convert image into tensor
    let tensor = tf.browser.fromPixels(img);

    // Resize image to 64 x 64
    tensor = tf.image.resizeBilinear(tensor, [IMAGE_SIZE, IMAGE_SIZE]);

    // Convert pixel values from 0-255 to 0-1
    tensor = tensor.div(255);

    return tensor;
  });
}

// Load all training images
async function createDataset() {
  const inputs = [];
  const outputs = [];

  // Plastic = 0
  for (const path of plasticImages) {
    const image = await loadImage(path);

    inputs.push(image);
    outputs.push(0);
  }

  // Metal = 1
  for (const path of metalImages) {
    const image = await loadImage(path);

    inputs.push(image);
    outputs.push(1);
  }

  return {
    inputs,
    outputs,
  };
}

// Create the CNN
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

  model.add(
    tf.layers.maxPooling2d({
      poolSize: 2,
      strides: 2,
    }),
  );

  model.add(
    tf.layers.conv2d({
      filters: 32,
      kernelSize: 3,
      padding: "same",
      activation: "relu",
    }),
  );

  model.add(
    tf.layers.maxPooling2d({
      poolSize: 2,
      strides: 2,
    }),
  );

  model.add(tf.layers.flatten());

  model.add(
    tf.layers.dense({
      units: 64,
      activation: "relu",
    }),
  );

  model.add(
    tf.layers.dense({
      units: 2,
      activation: "softmax",
    }),
  );

  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}

async function train() {
  console.log("Loading images...");

  const dataset = await createDataset();

  console.log("Images loaded:", dataset.inputs.length);

  // Combine all image tensors into one tensor
  const inputs = tf.stack(dataset.inputs);

  // Convert labels into one-hot vectors
  //
  // Plastic = [1, 0]
  // Metal   = [0, 1]
  const outputs = tf.oneHot(tf.tensor1d(dataset.outputs, "int32"), 2);

  console.log("Training data:", inputs.shape);
  console.log("Training labels:", outputs.shape);

  const model = createModel();

  model.summary();

  await model.fit(inputs, outputs, {
    epochs: 20,

    shuffle: true,

    batchSize: 4,

    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(
          `Epoch ${epoch + 1}:`,
          `loss = ${logs.loss.toFixed(4)}`,
          `accuracy = ${logs.acc?.toFixed(4) ?? logs.accuracy?.toFixed(4)}`,
        );
      },
    },
  });

  console.log("Training complete!");

  document.getElementById("prediction").innerText = "Training complete!";

  // Test the model
  await testModel(model, dataset);

  inputs.dispose();
  outputs.dispose();

  dataset.inputs.forEach((tensor) => tensor.dispose());
}

async function testModel(model, dataset) {
  // Pick a random image
  const index = Math.floor(Math.random() * dataset.inputs.length);

  const image = dataset.inputs[index];

  const prediction = tf.tidy(() => {
    return model.predict(image.expandDims(0));
  });

  const probabilities = await prediction.data();

  const predictedIndex = probabilities[0] > probabilities[1] ? 0 : 1;

  const confidence = probabilities[predictedIndex] * 100;

  document.getElementById("prediction").innerText =
    `${CLASSES[predictedIndex]} (${confidence.toFixed(1)}%)`;

  console.log("Actual:", CLASSES[dataset.outputs[index]]);

  console.log("Prediction:", CLASSES[predictedIndex]);

  console.log("Probabilities:", probabilities);

  prediction.dispose();
}

train();
