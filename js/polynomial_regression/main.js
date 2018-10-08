// Get a best fit line for a 3rd-order polynomial func
const pointsX = []
const pointsY = []
const canvasW = 480
const canvasH = 480

const learningRate = 0.1
const optimizer = tf.train.adamax(learningRate)

const a = tf.scalar(Math.random()).variable()
const b = tf.scalar(Math.random()).variable()
const c = tf.scalar(Math.random()).variable()
const d = tf.scalar(Math.random()).variable()

// loss function
// same as pred.sub(label).square().mean()
const loss = (pred, label) => tf.losses.meanSquaredError(label, pred)

// get predicted Y values
const predY = (inputX) => f(inputX).dataSync()

// polynomial function
// y = ax^3 + bx^2 + cx + d
const f = (x) => a.mul(x.pow(3))
  .add(b.mul(x.pow(2)))
  .add(c.mul(x.pow(1)))
  .add(d)

const mapCanvasToData = (x, y) => [
  map(x, 0, canvasW, -1, 1),
  map(y, 0, canvasH, 1, -1),
]

const mapDataToCanvas = (x, y) => [
  map(x, -1, 1, 0, canvasW),
  map(y, -1, 1, canvasH, 0),
]

const getTensors = () =>[
  tf.tensor1d(pointsX),
  tf.tensor1d(pointsY),
]

// optimize function (where the magic happens)
function optimize() {
  if (hasNoData()) return

  const [x, y] = getTensors()
  optimizer.minimize(() => loss(f(x), y))
}

const isOutOfRange = () => {
  const isOutOfRangeX = mouseX < 0 || mouseX > canvasW
  const isOutOfRangeY = mouseY < 0 || mouseY > canvasH

  return isOutOfRangeX || isOutOfRangeY
}

const hasNoData = () => {
  return pointsX.length === 0
}

// Get predicted curve
const getPredictedCurve = () => {
  // generate set of X values
  const curveX = []
  for (let i = -1; i <= 1; i += 0.01) curveX.push(i)

  // get predicted Y values
  const predictedX = tf.tensor1d(curveX)
  const curveY = tf.tidy(() => predY(predictedX))

  // clean up
  tf.dispose(predictedX)

  return [curveX, curveY]
}

const drawCurve = (curveX, curveY) => {
  beginShape()
  noFill()
  stroke(0)
  strokeWeight(2)
  for (let i = 0; i < curveX.length; i += 1) {
    vertex(...mapDataToCanvas(curveX[i], curveY[i]))
  }
  endShape()
}

const drawPoints = () => {
  strokeWeight(6)

  for (let i = 0; i < pointsX.length; i += 1) {
    point(...mapDataToCanvas(pointsX[i], pointsY[i]))
  }
}

function setup() {
  const el = createCanvas(canvasW, canvasH)
  el.parent('canvas')
}

function mouseClicked() {
  if (isOutOfRange()) return

  // add point to data array
  const [dataX, dataY] = mapCanvasToData(mouseX, mouseY)
  pointsX.push(dataX)
  pointsY.push(dataY)
}

function draw() {
  // run TensorFlow optimizer
  tf.tidy(optimize)

  // clear canvas
  clear()

  // draw points
  drawPoints()

  // draw curve
  drawCurve(...getPredictedCurve())
  
  // make sure no memory leak occurs
  console.log(tf.memory().numTensors)
}