// Animate the learning of XOR
const resolution = 50
const canvasW = 500
const canvasH = 500
const maxLimit = 1
const minLimit = 0
const learningRate = 0.02

const canvasInput = []
const model = tf.sequential()

const trainingData = [
  [[1, 0], 1],
  [[0, 1], 1],
  [[0, 0], 0],
  [[1, 1], 0],
]

const trainingInput = trainingData.map(data => data[0])
const trainingOutput = trainingData.map(data => data[1])

const inputTensor = tf.tensor2d(trainingInput)
const outputTensor = tf.tensor1d(trainingOutput)

const composeInput = () => {
  const cellRes = (maxLimit - minLimit) / resolution

  for (let j = minLimit; j <= maxLimit; j += cellRes) {
    for (let i = minLimit; i <= maxLimit; i += cellRes) {
      canvasInput.push([
        Math.round(i * 100) / 100, 
        Math.round(j * 100) / 100,
      ])
    }
  }
}

const drawGrid = async () => {
  const cellW = Math.floor(canvasW / resolution)
  const cellH = Math.floor(canvasH / resolution)

  const canvasOutputTensor = tf.tidy(getPredictions)
  const canvasOutput = await canvasOutputTensor.data()
  canvasOutputTensor.dispose()

  canvasInput.forEach(([x, y], i) => {
    xCoor = map(x, minLimit, maxLimit, 0, canvasW)
    yCoor = map(y, minLimit, maxLimit, 0, canvasH)

    noStroke()
    fill(canvasOutput[i] * 255)
    rect(xCoor, yCoor, cellW, cellH)
  })
}

const initTf = () => {
  // Defining model layers
  const hiddenLayer = tf.layers.dense({
    units: 4,
    activation: 'sigmoid',
    inputShape: [2],
  })

  const outputLayer = tf.layers.dense({
    units: 1,
    activation: 'sigmoid',
  })

  model.add(hiddenLayer)
  model.add(outputLayer)

  // Compile model
  model.compile({
    loss: 'meanSquaredError',
    optimizer: tf.train.adam(learningRate),
  })
}

const getPredictions = () => {
  const canvasInputTensor = tf.tensor2d(canvasInput)

  return model.predict(canvasInputTensor)
}

let isTraining = false
const lossEl = document.querySelector('#loss')

const trainModel = () => {
  isTraining = true

  return model.fit(inputTensor, outputTensor, {
    shuffle: true,
    epochs: 50,
  })
}

function setup() {
  const el = createCanvas(canvasW, canvasH)
  el.parent('canvas')

  composeInput()
  initTf()

  setInterval(drawGrid, 200)
}

function draw() {
  if (!isTraining) {
    tf.tidy(() => {
      trainModel().then((h) => {
        isTraining = false
        const loss = h.history.loss[0]
        console.log(loss)
        lossEl.innerHTML = loss
      })
    })
  }
  // drawGrid()
  // make sure no memory leak occurs
  // console.log(tf.memory().numTensors)
}