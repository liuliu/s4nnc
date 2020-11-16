import nnc
import Foundation

func DawnLayer(filters: Int, strides: Int, residual: Bool) -> Model {
  let input = Input()
  let conv = Model([
    Convolution(groups: 1, filters: filters, filterSize: [3, 3], noBias: false, hint: Hint(stride: [1, 1], border: Hint.Border([1, 1]))),
    BatchNorm(momentum: 0.9, epsilon: 1e-4),
    RELU()
  ])
  var output = conv(input)
  let pool = MaxPool(filterSize: [strides, strides], hint: Hint(stride: [strides, strides]))
  output = pool(output)
  if residual {
    let shortcut = output
    let res1 = Model([
      Convolution(groups: 1, filters: filters, filterSize: [3, 3], noBias: false, hint: Hint(stride: [1, 1], border: Hint.Border([1, 1]))),
      BatchNorm(momentum: 0.9, epsilon: 1e-4),
      RELU()
    ])
    output = res1(output)
    let res2 = Model([
      Convolution(groups: 1, filters: filters, filterSize: [3, 3], noBias: false, hint: Hint(stride: [1, 1], border: Hint.Border([1, 1]))),
      BatchNorm(momentum: 0.9, epsilon: 1e-4),
      RELU()
    ])
    output = res2(output)
    output = output .+ shortcut
  }
  return Model([input], [output])
}

func CIFAR10Dawn() -> Model {
  let prep = Model([
    Convolution(groups: 1, filters: 64, filterSize: [3, 3], noBias: false, hint: Hint(stride: [1, 1], border: Hint.Border([1, 1]))),
    BatchNorm(momentum: 0.9, epsilon: 1e-4),
    RELU()
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
    Dense(count: 10)
  ])
}

let graph = DynamicGraph()

let dataBatchPath = "/fast/Data/cifar-10/cifar-10-batches-bin/data_batch.bin"
let testBatchPath = "/fast/Data/cifar-10/cifar-10-batches-bin/test_batch.bin"

let dataBatch = try! Data(contentsOf: URL(fileURLWithPath: dataBatchPath))
let testBatch = try! Data(contentsOf: URL(fileURLWithPath: testBatchPath))

let dataBatchSize = 50_000
let testBatchSize = 10_000

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
print(mean)
