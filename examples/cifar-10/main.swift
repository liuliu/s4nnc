import Foundation
import NNC

func DawnLayer(filters: Int, strides: Int, residual: Bool) -> Model {
  let input = Input()
  let conv = Model([
    Convolution(
      groups: 1, filters: filters, filterSize: [3, 3], noBias: false,
      hint: Hint(stride: [1, 1], border: Hint.Border([1, 1]))),
    BatchNorm(momentum: 0.9, epsilon: 1e-4),
    ReLU(),
  ])
  var output = conv(input)
  let pool = MaxPool(filterSize: [strides, strides], hint: Hint(stride: [strides, strides]))
  output = pool(output)
  if residual {
    let shortcut = output
    let res1 = Model([
      Convolution(
        groups: 1, filters: filters, filterSize: [3, 3], noBias: false,
        hint: Hint(stride: [1, 1], border: Hint.Border([1, 1]))),
      BatchNorm(momentum: 0.9, epsilon: 1e-4),
      ReLU(),
    ])
    output = res1(output)
    let res2 = Model([
      Convolution(
        groups: 1, filters: filters, filterSize: [3, 3], noBias: false,
        hint: Hint(stride: [1, 1], border: Hint.Border([1, 1]))),
      BatchNorm(momentum: 0.9, epsilon: 1e-4),
      ReLU(),
    ])
    output = res2(output)
    output = output .+ shortcut
  }
  return Model([input], [output])
}

func CIFAR10Dawn() -> Model {
  let prep = Model([
    Convolution(
      groups: 1, filters: 64, filterSize: [3, 3], noBias: false,
      hint: Hint(stride: [1, 1], border: Hint.Border([1, 1]))),
    BatchNorm(momentum: 0.9, epsilon: 1e-4),
    ReLU(),
  ])
  let layer1 = DawnLayer(filters: 128, strides: 2, residual: true)
  let layer2 = DawnLayer(filters: 256, strides: 2, residual: false)
  let layer3 = DawnLayer(filters: 512, strides: 2, residual: true)
  return Model([
    prep,
    layer1,
    layer2,
    layer3,
    MaxPool(filterSize: [0, 0], hint: Hint()),
    Flatten(),
    Dense(count: 10),
  ])
}

/// MARK - The Training Program

let dataBatchPath = "/fast/Data/cifar-10/cifar-10-batches-bin/data_batch.bin"
let testBatchPath = "/fast/Data/cifar-10/cifar-10-batches-bin/test_batch.bin"

let dataBatchSize = 50_000
let testBatchSize = 10_000

let batchSize = 1024

/// MARK - Loading Data from Disk

let dataBatch = try! Data(contentsOf: URL(fileURLWithPath: dataBatchPath))
let testBatch = try! Data(contentsOf: URL(fileURLWithPath: testBatchPath))

struct CIFARData {
  var tensor: Tensor<Float32>
  var label: Int
  var mean: (Float, Float, Float)
}

var trainData = [CIFARData?](repeating: nil, count: dataBatchSize)

DispatchQueue.concurrentPerform(iterations: dataBatchSize) { k in
  var tensor = Tensor<Float32>(.CPU, .HWC(32, 32, 3))
  let label = Int(dataBatch[k * (3 * 32 * 32 + 1)])
  let imageData = dataBatch.subdata(in: (k * (3 * 32 * 32 + 1) + 1)..<((k + 1) * (3 * 32 * 32 + 1)))
  var mean: (Float, Float, Float) = (0.0, 0.0, 0.0)
  for i in 0..<32 {
    for j in 0..<32 {
      let r = Float(imageData[i * 32 + j]) * 2.0 / 255.0
      let g = Float(imageData[32 * 32 + i * 32 + j]) * 2.0 / 255.0
      let b = Float(imageData[32 * 32 * 2 + i * 32 + j]) * 2.0 / 255.0
      tensor[i, j, 0] = r
      tensor[i, j, 1] = g
      tensor[i, j, 2] = b
      mean.0 += r
      mean.1 += g
      mean.2 += b
    }
  }
  mean.0 = mean.0 / Float(32 * 32)
  mean.1 = mean.1 / Float(32 * 32)
  mean.2 = mean.2 / Float(32 * 32)
  trainData[k] = CIFARData(tensor: tensor, label: label, mean: mean)
}
var mean: (Double, Double, Double) = (0.0, 0.0, 0.0)
for k in 0..<dataBatchSize {
  guard let cifarData = trainData[k] else {
    fatalError("something is wrong with the trainData array")
  }
  mean.0 += Double(cifarData.mean.0)
  mean.1 += Double(cifarData.mean.1)
  mean.2 += Double(cifarData.mean.2)
}
mean.0 = mean.0 / Double(dataBatchSize)
mean.1 = mean.1 / Double(dataBatchSize)
mean.2 = mean.2 / Double(dataBatchSize)

var testData = [CIFARData?](repeating: nil, count: testBatchSize)
let meanf: (Float, Float, Float) = (Float(mean.0), Float(mean.1), Float(mean.2))

DispatchQueue.concurrentPerform(iterations: testBatchSize) { k in
  var tensor = Tensor<Float32>(.CPU, .HWC(32, 32, 3))
  let label = Int(dataBatch[k * (3 * 32 * 32 + 1)])
  let imageData = dataBatch.subdata(in: (k * (3 * 32 * 32 + 1) + 1)..<((k + 1) * (3 * 32 * 32 + 1)))
  for i in 0..<32 {
    for j in 0..<32 {
      tensor[i, j, 0] = Float(imageData[i * 32 + j]) * 2.0 / 255.0 - meanf.0
      tensor[i, j, 1] = Float(imageData[32 * 32 + i * 32 + j]) * 2.0 / 255.0 - meanf.1
      tensor[i, j, 2] = Float(imageData[32 * 32 * 2 + i * 32 + j]) * 2.0 / 255.0 - meanf.2
    }
  }
  testData[k] = CIFARData(tensor: tensor, label: label, mean: meanf)
}

/// MARK - Setup Data Feeder Pipelne

var trainDataDf = DataFrame(from: trainData, name: "main")
let testDataDf = DataFrame(from: testData, name: "main")
trainDataDf["tensor"] = trainDataDf["main", CIFARData.self].map(\.tensor)
trainDataDf["c"] = trainDataDf["main", CIFARData.self].map {
  Tensor<Int32>([Int32($0.label)], .CPU, .C(1))
}
trainDataDf["jittered"] = trainDataDf["tensor"]!.toImageJitter(
  Float32.self,
  size: ImageJitter.Size(rows: 32, cols: 32),
  resize: ImageJitter.Resize(min: 32, max: 32),
  symmetric: true,
  offset: ImageJitter.Offset(x: 4, y: 4),
  normalize: ImageJitter.Normalize(mean: [meanf.0, meanf.1, meanf.2])
)

var batchedTrainData = trainDataDf["jittered", "c"].combine(size: batchSize)
let toGPUTrain = batchedTrainData["jittered", "c"].toGPU()
batchedTrainData["jitteredGPU"] = toGPUTrain["jittered"]
batchedTrainData["cGPU"] = toGPUTrain["c"]

/// MARK - Training Loop

let graph = DynamicGraph()
let cifar = CIFAR10Dawn()
var overallAccuracy = 0.0
var sgd0 = SGDOptimizer(
  graph, nesterov: true, rate: 0.0001, scale: 1.0 / Float(batchSize), decay: 0.001, momentum: 0.9,
  dampening: 0)
sgd0.parameters = [cifar.parameters]
var sgd1 = SGDOptimizer(
  graph, nesterov: true, rate: 0.0001, scale: 1.0 / Float(batchSize), decay: 0, momentum: 0.9,
  dampening: 0)
sgd1.parameters = [cifar.parameters(for: .bias)]
var optimizers = [sgd0, sgd1]
for epoch in 0..<35 {
  batchedTrainData.shuffle()
  for (i, batch) in batchedTrainData["jitteredGPU", "cGPU"].enumerated() {
    // Piece-wise linear learning rate: https://www.myrtle.ai/2018/09/24/how_to_train_your_resnet_3/
    let overallIndex = i + epoch * batchedTrainData.count
    var learnRate: Float
    if overallIndex + 1 < 5 * batchedTrainData.count {
      learnRate = 0.4 * Float(overallIndex + 1) / Float(5 * batchedTrainData.count)
    } else {
      learnRate =
        0.4 * Float(30 * batchedTrainData.count - (overallIndex + 1))
        / Float((30 - 5) * batchedTrainData.count)
    }
    learnRate = max(learnRate, 0.0001)
    optimizers[0].rate = learnRate
    optimizers[1].rate = learnRate
    let tensorGPU = batch[0] as! Tensor<Float32>
    let cGPU = batch[1] as! Tensor<Int32>
    let input = graph.variable(tensorGPU)
    let output = cifar(inputs: input)[0]
    let softmaxLoss = SoftmaxCrossEntropyLoss()
    let target = graph.constant(cGPU)
    let loss = softmaxLoss(output, target: target)
    loss.backward(to: [input])
    optimizers.step()
    let c = cGPU.toCPU()
    var correct = 0
    let cpuOutput = DynamicGraph.Tensor<Float32>(output).toCPU()
    for i in 0..<batchSize {
      let label = c[i, 0]
      var prediction: Int = 0
      var predScore: Float = cpuOutput[i, 0]
      for j in 1..<10 {
        if cpuOutput[i, j] > predScore {
          prediction = j
          predScore = cpuOutput[i, j]
        }
      }
      if label == prediction {
        correct += 1
      }
    }
    let accuracy = Double(correct) / Double(batchSize)
    overallAccuracy = overallAccuracy * 0.9 + accuracy * 0.1
    if (i + 1) % 50 == 0 {
      print("epoch \(epoch) (\(i)/\(batchedTrainData.count)), training accuracy \(overallAccuracy)")
    }
  }
}
