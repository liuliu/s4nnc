import Foundation
import NNC
import TensorBoard

public func timeEmbedding(timesteps: [Float], embeddingSize: Int, maxPeriod: Int)
  -> Tensor<
    Float
  >
{
  precondition(embeddingSize % 2 == 0)
  var embedding = Tensor<Float>(.CPU, .NC(timesteps.count, embeddingSize))
  let half = embeddingSize / 2
  for j in 0..<timesteps.count {
    let timestep = timesteps[j]
    for i in 0..<half {
      let freq: Float = exp(-log(Float(maxPeriod)) * Float(i) / Float(half)) * timestep
      let cosFreq = cos(freq)
      let sinFreq = sin(freq)
      embedding[j, i] = cosFreq
      embedding[j, i + half] = sinFreq
    }
  }
  return embedding
}

public func TimestepEmbedder(hiddenSize: Int) -> Model {
  let x = Input()
  let fc0 = Dense(count: hiddenSize, name: "timestep_embedder_0")
  var out = fc0(x).swish()
  let fc2 = Dense(count: hiddenSize, name: "timestep_embedder_1")
  out = fc2(out)
  return Model([x], [out])
}

public func LabelEmbedder<T: TensorNumeric>(_ dataType: T.Type, numClasses: Int, hiddenSize: Int)
  -> Model
{
  let labelEmbed = Embedding(
    T.self, vocabularySize: numClasses + 1, embeddingSize: hiddenSize, name: "label_embedder")
  return labelEmbed
}

func SelfAttention(prefix: String, k: Int, h: Int, hk: Int, b: Int, t: Int) -> Model {
  let x = Input()
  let rot = Input()
  let tokeys = Dense(count: k * hk, noBias: true, name: "k_proj")
  let toqueries = Dense(count: k * h, noBias: true, name: "q_proj")
  let tovalues = Dense(count: k * hk, noBias: true, name: "v_proj")
  let k_norm = LayerNorm(epsilon: 1e-6, axis: [2], name: "k_norm")
  var keys = k_norm(tokeys(x)).reshaped([b, t, hk, k])
  let q_norm = LayerNorm(epsilon: 1e-6, axis: [2], name: "q_norm")
  var queries = q_norm(toqueries(x)).reshaped([b, t, h, k])
  let values = tovalues(x).reshaped([b, t, hk, k]).transposed(1, 2)
  keys = Functional.cmul(left: keys, right: rot)
  keys = keys.transposed(1, 2)
  queries = Functional.cmul(left: queries, right: rot)
  queries = ((1.0 / Float(k).squareRoot()) * queries).transposed(1, 2)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * t, t])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, t, t])
  var out = dot * values
  out = out.reshaped([b, h, t, k]).transposed(1, 2).reshaped([b, t, h * k])
  let unifyheads = Dense(count: k * h, noBias: true, name: "out_proj")
  out = unifyheads(out)
  return Model([x, rot], [out])
}

func FeedForward(hiddenSize: Int, intermediateSize: Int, name: String = "") -> Model {
  let x = Input()
  let w1 = Dense(count: intermediateSize, noBias: true, name: "ff_w1")
  let w3 = Dense(count: intermediateSize, noBias: true, name: "ff_w3")
  var out = w3(x) .* w1(x).swish()
  let w2 = Dense(count: hiddenSize, noBias: true, name: "ff_w2")
  out = w2(out)
  return Model([x], [out], name: name)
}

func TransformerBlock(k: Int, h: Int, hk: Int, b: Int, t: Int) -> Model {
  let x = Input()
  let rot = Input()
  let y = Input()
  let adaLNs = (0..<6).map { Dense(count: k * h, name: "ada_ln_\($0)") }
  let chunks = adaLNs.map { $0(y) }
  let attention = SelfAttention(prefix: "", k: k, h: h, hk: hk, b: b, t: t)
  let attentionNorm = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  var out = x + chunks[2] .* attention(attentionNorm(x) .* (1 + chunks[1]) + chunks[0], rot)
  let ffn = FeedForward(hiddenSize: k * h, intermediateSize: k * h * 3)
  let ffnNorm = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  out = out + chunks[5] .* ffn(ffnNorm(out) .* (1 + chunks[4]) + chunks[3])
  return Model([x, rot, y], [out])
}

func DiT(batchSize: Int, hiddenSize: Int, layers: Int) -> Model {
  let x = Input()
  let conv0 = Convolution(
    groups: 1, filters: hiddenSize / 2, filterSize: [5, 5],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [2, 2], end: [2, 2])), name: "conv0")
  let norm0 = GroupNorm(axis: 1, groups: 32, epsilon: 1e-5, reduce: [2, 3], name: "norm0")
  var out = norm0(conv0(x).swish())
  let conv1 = Convolution(
    groups: 1, filters: hiddenSize / 2, filterSize: [5, 5],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [2, 2], end: [2, 2])), name: "conv1")
  let norm1 = GroupNorm(axis: 1, groups: 32, epsilon: 1e-5, reduce: [2, 3], name: "norm1")
  out = norm1(conv1(out).swish())
  out = out.reshaped([batchSize, hiddenSize / 2, 16, 2, 16, 2]).permuted(0, 2, 4, 3, 5, 1)
    .contiguous().reshaped([batchSize, 16 * 16, hiddenSize * 2])
  let xEmbedder = Dense(count: hiddenSize, name: "x_embedder")
  out = xEmbedder(out)

  let rot = Input()
  let t = Input()
  let timestepEmbedder = TimestepEmbedder(hiddenSize: hiddenSize)
  let y = Input()
  let labelEmbedder = LabelEmbedder(Float.self, numClasses: 10, hiddenSize: hiddenSize)
  let adaLNInput = (timestepEmbedder(t) + labelEmbedder(y)).reshaped([batchSize, 1, hiddenSize])
    .swish()
  for _ in 0..<layers {
    let transformer = TransformerBlock(k: hiddenSize / 8, h: 8, hk: 8, b: batchSize, t: 256)
    out = transformer(out, rot, adaLNInput)
  }
  let norm = LayerNorm(epsilon: 1e-6, axis: [2], elementwiseAffine: false)
  out = norm(out)
  let adaLNs = [
    Dense(count: hiddenSize, name: "ada_ln_final_0"),
    Dense(count: hiddenSize, name: "ada_ln_final_1"),
  ]
  let chunks = adaLNs.map { $0(adaLNInput) }
  out = out .* (1 + chunks[1]) + chunks[0]
  let convOut = Dense(count: 3 * 2 * 2, name: "final")
  out = convOut(out).reshaped([batchSize, 16, 16, 2, 2, 3]).permuted(0, 5, 1, 3, 2, 4).contiguous()
    .reshaped([batchSize, 3, 32, 32])
  return Model([x, rot, t, y], [out])
}

/// MARK - The Training Program

let dataBatchPath = "/fast/Data/cifar-10/cifar-10-batches-bin/data_batch.bin"

let dataBatchSize = 50_000

let globalBatchSize = 256

/// MARK - Loading Data from Disk

let dataBatch = try Data(contentsOf: URL(fileURLWithPath: dataBatchPath))

struct CIFARData {
  var tensor: Tensor<Float>
  var label: Int
}

var trainData = [CIFARData?](repeating: nil, count: dataBatchSize)

DispatchQueue.concurrentPerform(iterations: dataBatchSize) { k in
  var tensor = Tensor<Float>(.CPU, .HWC(32, 32, 3))
  let label = Int(dataBatch[k * (3 * 32 * 32 + 1)])
  let imageData = dataBatch.subdata(in: (k * (3 * 32 * 32 + 1) + 1)..<((k + 1) * (3 * 32 * 32 + 1)))
  for i in 0..<32 {
    for j in 0..<32 {
      let r = Float(imageData[i * 32 + j]) * 2.0 / 255.0 - 1.0
      let g = Float(imageData[32 * 32 + i * 32 + j]) * 2.0 / 255.0 - 1.0
      let b = Float(imageData[32 * 32 * 2 + i * 32 + j]) * 2.0 / 255.0 - 1.0
      tensor[i, j, 0] = r
      tensor[i, j, 1] = g
      tensor[i, j, 2] = b
    }
  }
  trainData[k] = CIFARData(tensor: tensor, label: label)
}

/// MARK - Setup Data Feeder Pipelne

var trainDataDf = DataFrame(from: trainData, name: "main")
trainDataDf["tensor"] = trainDataDf["main", CIFARData.self].map(\.tensor)
trainDataDf["c"] = trainDataDf["main", CIFARData.self].map {
  Tensor<Int32>([Int32($0.label)], .CPU, .C(1))
}

let deviceCount = DeviceKind.GPUs.count
let batchSize = globalBatchSize / deviceCount
var batchedTrainData = trainDataDf["tensor", "c"].combine(size: batchSize, repeating: deviceCount)
if deviceCount > 1 {
  for i in 0..<deviceCount {
    batchedTrainData["imageGPU_\(i)"] = batchedTrainData["tensor_\(i)"]!.toGPU(i)
  }
} else {
  batchedTrainData["imageGPU"] = batchedTrainData["tensor"]!.toGPU(0)
}

/// MARK - Training Loop

let summaryWriter = SummaryWriter(logDirectory: "/tmp/minrf")

let graph = DynamicGraph()
let dit = DiT(batchSize: batchSize, hiddenSize: 256, layers: 10)
var rot = graph.variable(.CPU, .NCHW(batchSize, 16 * 16, 8, 32), of: Float.self)
for i in 0..<(16 * 16) {
  for k in 0..<16 {
    let theta = Double(i) * 1.0 / pow(10_000, Double(k) * 2 / 32)
    let sintheta = sin(theta)
    let costheta = cos(theta)
    for b in 0..<batchSize {
      for h in 0..<8 {
        rot[b, i, h, k * 2] = Float(costheta)
        rot[b, i, h, k * 2 + 1] = Float(sintheta)
      }
    }
  }
}
let rotG = DynamicGraph.Group((0..<deviceCount).map { rot.toGPU($0) })
var optimizer = AdamWOptimizer(graph, rate: 0.0005)
optimizer.parameters = [dit.parameters]
var isLoaded = false
var columns = [String]()
if deviceCount > 1 {
  for i in 0..<deviceCount {
    columns += ["imageGPU_\(i)", "c_\(i)"]
  }
} else {
  columns += ["imageGPU", "c"]
}
for epoch in 0..<10000 {
  batchedTrainData.shuffle()
  var overallLoss: Double = 0
  var overallSamples = 0
  var lossCount: [Int] = Array(repeating: 0, count: 10)
  var lossBins: [Double] = Array(repeating: 0, count: 10)
  for (_, batch) in batchedTrainData[columns].enumerated() {
    let x = graph.variable((0..<deviceCount).map { batch[$0 * 2] as! Tensor<Float> })
    let y = graph.variable(
      (0..<deviceCount).map { (batch[$0 * 2 + 1] as! Tensor<Int32>).reshaped(.C(batchSize)) })
    let labelDrop = graph.variable(.CPU, .NC(deviceCount, batchSize), of: Float.self)
    labelDrop.rand()
    for i in 0..<deviceCount {
      for j in 0..<batchSize {
        // 10% chance of dropping the label.
        if labelDrop[i, j] < 0.1 {
          y[i][j] = 10
        }
      }
    }
    let yG = DynamicGraph.Group(y.enumerated().map { $1.toGPU($0) })
    let t = DynamicGraph.Group(
      (0..<deviceCount).map { _ in graph.variable(.CPU, .C(batchSize), of: Float.self) })
    t.randn()
    t.sigmoid()
    let tG = DynamicGraph.Group(t.enumerated().map { $1.toGPU($0) }).reshaped(
      .NCHW(batchSize, 1, 1, 1))
    let tE = graph.variable(
      (0..<deviceCount).map { i in
        timeEmbedding(
          timesteps: (0..<batchSize).map({ t[i][$0] }), embeddingSize: 256, maxPeriod: 10_000)
      })
    let tEG = DynamicGraph.Group(tE.enumerated().map { $1.toGPU($0) })
    let z1 = graph.variable(like: x)
    z1.randn()
    let zt = (1 - tG) .* x + tG .* z1
    if !isLoaded {
      dit.compile(inputs: zt, rotG, tEG, yG, isEager: true)
      if let final = dit.parameters.first(where: { $0.contains("final-0") }) {
        let weight = graph.variable(final.copied(Float.self))
        weight.full(0)
        final.copy(from: weight)
      }
      isLoaded = true
    }
    let vtheta = dit(inputs: zt, rotG, tEG, yG)[0].as(of: Float.self)
    let diff = z1 - x - vtheta
    let loss = (diff .* diff).reduced(.mean, axis: [1, 2, 3])
    loss.backward(to: [zt])
    optimizer.step()
    let lossC = loss.map { $0.rawValue.toCPU() }
    var batchLoss: Double = 0
    for i in 0..<deviceCount {
      for j in 0..<batchSize {
        let singleLoss = Double(lossC[i][j, 0, 0, 0])
        lossBins[min(Int((t[i][j] * 10).rounded(.down)), 9)] += singleLoss
        lossCount[min(Int((t[i][j] * 10).rounded(.down)), 9)] += 1
        batchLoss += singleLoss
      }
    }
    batchLoss = batchLoss / Double(batchSize * deviceCount)
    overallLoss += batchLoss
    overallSamples += 1
    summaryWriter.addScalar("loss", batchLoss, step: optimizer.step)
  }
  overallLoss = overallLoss / Double(overallSamples)
  print("overall loss \(overallLoss), epoch \(epoch)")
  summaryWriter.addScalar("overall loss", overallLoss, step: epoch)

  for i in 0..<10 {
    let loss = lossBins[i] / Double(lossCount[i])
    summaryWriter.addScalar("lossbin \(i)", loss, step: epoch)
  }
  summaryWriter.addParameters("parameters", dit.parameters, step: epoch)
  // Run denoising.
  let samplingSteps = 50
  graph.withNoGrad {
    var z = DynamicGraph.Group(
      (0..<deviceCount).map {
        graph.variable(.GPU($0), .NCHW(batchSize, 3, 32, 32), of: Float.self)
      })
    z.randn()
    let y = graph.variable(.CPU, .C(batchSize), of: Int32.self)
    for i in 0..<batchSize {
      y[i] = Int32(i) % 10
    }
    let yG = DynamicGraph.Group((0..<deviceCount).map { y.toGPU($0) })
    let u = graph.variable(.CPU, .C(batchSize), of: Int32.self)
    for i in 0..<batchSize {
      u[i] = 10
    }
    let uG = DynamicGraph.Group((0..<deviceCount).map { u.toGPU($0) })
    for i in (1...samplingSteps).reversed() {
      let t = Float(i) / Float(samplingSteps)
      let tE = graph.variable(
        timeEmbedding(
          timesteps: (0..<batchSize).map({ _ in t }), embeddingSize: 256, maxPeriod: 10_000))
      let tEG = DynamicGraph.Group((0..<deviceCount).map { tE.toGPU($0) })
      let vc = dit(inputs: z, rotG, tEG, yG)[0].as(of: Float.self)
      let vu = dit(inputs: z, rotG, tEG, uG)[0].as(of: Float.self)
      // cfg = 2
      let v = vu + 2 * (vc - vu)
      z = z - (1 / Float(samplingSteps)) * v
    }
    let zCPU = z.toCPU()
    for i in 0..<deviceCount {
      for j in 0..<batchSize {
        // Write each image as ppm format.
        summaryWriter.addImage(
          "sample \(i * batchSize + j)", (zCPU[i][j..<(j + 1), 0..<3, 0..<32, 0..<32] + 1) * 0.5,
          step: epoch)
      }
    }
  }
}
