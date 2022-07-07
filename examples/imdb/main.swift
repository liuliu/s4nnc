import Foundation
import NNC

/// MARK - Setup the Transformer Model

func SelfAttention(k: Int, h: Int, b: Int, t: Int, dropout: Float) -> Model {
  let x = Input()
  let mask = Input()
  let multiheads = x.reshaped([b * t, k])
  let tokeys = Dense(count: k * h, noBias: true)
  let toqueries = Dense(count: k * h, noBias: true)
  let tovalues = Dense(count: k * h, noBias: true)
  let keys = tokeys(multiheads).reshaped([t, b, h, k]).transposed(0, 2).reshaped([b * h, t, k])
  let queries = toqueries(multiheads).reshaped([t, b, h, k]).transposed(0, 2).reshaped([
    b * h, t, k,
  ])
  let values = tovalues(multiheads).reshaped([t, b, h, k]).transposed(0, 2).reshaped([b * h, t, k])
  var dot = Matmul(transposeB: (1, 2))(queries, keys)
  dot = (1.0 / Float(k).squareRoot()) * dot
  dot = MaskedFill(equalTo: 0, fillWith: 1e-9)(dot, mask)
  dot = dot.reshaped([b * h * t, t])
  dot = Softmax()(dot)
  if dropout > 0 {
    dot = Dropout(probability: dropout)(dot)
  }
  dot = dot.reshaped([b * h, t, t])
  var out = dot * values
  out = out.reshaped([h, b, t, k]).transposed(0, 2).reshaped([b * t, h * k])
  let unifyheads = Dense(count: k)
  out = unifyheads(out).reshaped([t, b, k])
  return Model([x, mask], [out])
}

func TransformerBlock(k: Int, h: Int, b: Int, t: Int, ff: Int, dropout: Float) -> Model {
  let x = Input()
  let mask = Input()
  let selfAttention = SelfAttention(k: k, h: h, b: b, t: t, dropout: dropout)
  var out = selfAttention(x, mask)
  out = x + out
  let first = LayerNorm(epsilon: 1e-5, axis: [2])(out)
  if dropout > 0 {
    out = Dropout(probability: dropout)(first)
  } else {
    out = first
  }
  out = out.reshaped([b * t, k])
  out = Dense(count: ff)(out)
  out = ReLU()(out)
  out = Dense(count: k)(out)
  out = out.reshaped([t, b, k])
  out = first + out
  out = LayerNorm(epsilon: 1e-5, axis: [2])(out)
  if dropout > 0 {
    out = Dropout(probability: dropout)(out)
  }
  return Model([x, mask], [out])
}

func ClassicTransformer(layers: Int, k: Int, h: Int, b: Int, t: Int, ff: Int, dropout: Float)
  -> Model
{
  let x = Input()
  let mask = Input()
  var out = x.transposed(0, 1)
  for _ in 0..<layers {
    out = TransformerBlock(k: k, h: h, b: b, t: t, ff: ff, dropout: dropout)(out, mask)
  }
  out = out.transposed(0, 1).transposed(1, 2).reshaped([b, k, t, 1])
  out = AveragePool()(out)
  out = Flatten()(out)
  out = Dense(count: 2)(out)
  return Model([x, mask], [out])
}

struct TransformerParameter {
  var ff: Int
  var layers: Int
  var h: Int
  var dropout: Float
}

/// MARK - The Training Program

let parameters = TransformerParameter(ff: 4, layers: 2, h: 8, dropout: 0.1)

let transformer: ModelBuilder = ModelBuilder { inputs in
  let b = inputs[0].dimensions[0]
  let t = inputs[0].dimensions[1]
  let k = inputs[0].dimensions[2]
  return ClassicTransformer(
    layers: parameters.layers, k: k, h: parameters.h, b: b, t: t, ff: parameters.ff * k,
    dropout: parameters.dropout)
}

let trainListFile = "/fast/Data/IMDB_Movie_Reviews/aclImdb/train.txt"
let testListFile = "/fast/Data/IMDB_Movie_Reviews/aclImdb/test.txt"
let vocabFile = "/fast/Data/IMDB_Movie_Reviews/aclImdb/imdb.vocab"
let baseDir = "/fast/Data/IMDB_Movie_Reviews/aclImdb"

let vocabContent = try! String(contentsOfFile: vocabFile, encoding: .utf8)
let vocabList = vocabContent.split(separator: "\n")

var dict = [String: Int]()
for (i, word) in vocabList.enumerated() {
  let lowercasedWord = word.lowercased()
  dict[lowercasedWord] = i
}

/// MARK - Data Processing

let unknownFlag = Int32(vocabList.count)
let endFlag = Int32(vocabList.count + 1)
let padFlag = Int32(vocabList.count + 2)
let maxLength = 512
let vocabSize = vocabList.count + 3
let embeddingSize = 128
let batchSize = 64

struct ImdbText {
  var tensor: Tensor<Int32>
  var mask: Tensor<Int32>
  var c: Int
}

func dataFromDisk(filePath trainListFile: String) -> DataFrame {
  let trainListContent = try! String(contentsOfFile: trainListFile, encoding: .utf8)
  let trainList = trainListContent.split(separator: "\n")
  var trainData = [ImdbText]()
  for trainFile in trainList {
    let parts = trainFile.split(separator: " ")
    let c = Int(parts[0])!
    let filePath = parts[1...].joined(separator: " ")
    let trainText = try! String(contentsOfFile: "\(baseDir)/\(filePath)", encoding: .utf8)
    let lowercasedTrainText = trainText.lowercased()
    let separators: Set<Character> = [
      " ", ".", ",", "<", ">", "/", "~", "`", "@", "#", "$", "%", "^", "&", "*", "+", "\\", "\"",
    ]
    let tokens = lowercasedTrainText.split(whereSeparator: { character in
      return separators.contains(character)
    })
    var tensor = Tensor<Int32>(.CPU, .C(maxLength))
    for (i, token) in tokens.enumerated() where i < maxLength {
      tensor[i] = dict[String(token)].map { Int32($0) } ?? unknownFlag
    }
    if tokens.count < maxLength {
      for i in tokens.count..<maxLength {
        tensor[i] = i == tokens.count ? endFlag : padFlag
      }
    }
    var mask = Tensor<Int32>(.CPU, .C(1))
    mask[0] = Int32(min(tokens.count + 1, maxLength))
    let imdbText = ImdbText(tensor: tensor, mask: mask, c: c)
    trainData.append(imdbText)
  }
  return DataFrame(from: trainData, name: "main")
}

/// MARK - Setup the Data Feeder Pipeline

var trainData = dataFromDisk(filePath: trainListFile)
var testData = dataFromDisk(filePath: testListFile)
// Extract tensors from ImdbText struct.
trainData["tensor"] = trainData["main", ImdbText.self].map(\.tensor)
trainData["mask"] = trainData["main", ImdbText.self].map(\.mask)
trainData["c"] = trainData["main", ImdbText.self].map(\.c)
// Create one hot tensor out of the scalar.
trainData["oneHot"] = trainData["c", Int.self].toOneHot(Float32.self, count: 2)
// Do above for test data.
testData["tensor"] = testData["main", ImdbText.self].map(\.tensor)
testData["mask"] = testData["main", ImdbText.self].map(\.mask)
testData["c"] = testData["main", ImdbText.self].map(\.c)
testData["oneHot"] = testData["c", Int.self].toOneHot(Float32.self, count: 2)

let deviceCount = DeviceKind.GPUs.count

// Batching tensors together.
var batchedTrainData = trainData["tensor", "mask", "oneHot"].combine(
  size: batchSize, repeating: deviceCount)
for i in 0..<deviceCount {
  batchedTrainData["truncTensor_\(i)"] = batchedTrainData["tensor_\(i)"]!.toTruncate(
    batchedTrainData["mask_\(i)"]!)
  batchedTrainData["squaredMask_\(i)"] = batchedTrainData["mask_\(i)"]!.toOneSquared(
    maxLength: maxLength)
  // Move the tensors from CPU to GPU.
  let toGPUTrain = batchedTrainData["truncTensor_\(i)", "oneHot_\(i)", "squaredMask_\(i)"].toGPU(i)
  batchedTrainData["tensorGPU_\(i)"] = toGPUTrain["truncTensor_\(i)"]
  batchedTrainData["oneHotGPU_\(i)"] = toGPUTrain["oneHot_\(i)"]
  batchedTrainData["squaredMaskGPU_\(i)"] = toGPUTrain["squaredMask_\(i)"]
}

/// MARK - The Training Loop

let graph = DynamicGraph()

let vocabVec: DynamicGraph.Group<DynamicGraph.Tensor<Float32>> = DynamicGraph.Group(
  (0..<deviceCount).map { graph.variable(.GPU($0), .NC(vocabSize, embeddingSize)) })
let seqVec: DynamicGraph.Group<DynamicGraph.Tensor<Float32>> = DynamicGraph.Group(
  (0..<deviceCount).map { graph.variable(.GPU($0), .NC(maxLength, embeddingSize)) })
vocabVec.rand(-1...1)
seqVec.rand(-1...1)
graph.openStore("/home/liu/workspace/s4nnc/imdb.checkpoint") { store in
  store.read("vocab", variable: vocabVec)
  store.read("seq", variable: seqVec)
  store.read("transformer", model: transformer)
}
var adamOptimizer = AdamOptimizer(
  graph, rate: 0.0001, betas: (0.9, 0.98), decay: 0, epsilon: 1e-9)
adamOptimizer.parameters = [vocabVec, seqVec, transformer.parameters]
var overallAccuracy = 0.0
for epoch in 0..<10 {
  batchedTrainData.shuffle()
  var columns = [String]()
  for i in 0..<deviceCount {
    columns += ["tensorGPU_\(i)", "oneHotGPU_\(i)", "squaredMaskGPU_\(i)"]
  }
  let computeStream = StreamContext(.GPU(0))
  for (i, batch) in batchedTrainData[columns].enumerated() {
    adamOptimizer.rate =
      0.0001 * min(Float(adamOptimizer.step - 1) / (10000.0 / Float(batchSize)), 1)
      * Float(deviceCount)
    let tensorGPU = (0..<deviceCount).map { batch[$0 * 3] as! Tensor<Int32> }
    let oneHotGPU = (0..<deviceCount).map { batch[$0 * 3 + 1] as! Tensor<Float32> }
    let squaredMaskGPU = (0..<deviceCount).map { batch[$0 * 3 + 2] as! Tensor<Int32> }
    let batchLength = tensorGPU[0].dimensions[1]
    let output = graph.withStream(computeStream) {
      () -> DynamicGraph.Group<DynamicGraph.AnyTensor> in
      let wordIndices = graph.variable(tensorGPU.reshaped(.C(batchSize * batchLength)))
      let wordVec = Functional.indexSelect(input: vocabVec, index: wordIndices)
      var seqIndicesCPU = Tensor<Int32>(.CPU, .C(batchSize * batchLength))
      for i in 0..<batchSize {
        for j in 0..<batchLength {
          seqIndicesCPU[i * batchLength + j] = Int32(j)
        }
      }
      let seqIndicesGPU = (0..<deviceCount).map { seqIndicesCPU.toGPU($0) }
      let seqIndices = graph.constant(seqIndicesGPU)
      let posVec = Functional.indexSelect(input: seqVec, index: seqIndices)
      let selectVec = wordVec + posVec
      let inputVec = selectVec.reshaped(.CHW(batchSize, batchLength, embeddingSize))
      let masked = graph.constant(
        squaredMaskGPU.reshaped(.CHW(batchSize, batchLength, batchLength)))
      let output = transformer(inputs: inputVec, masked)[0]
      let softmaxLoss = SoftmaxCrossEntropyLoss()
      let target = graph.variable(oneHotGPU)
      let loss = softmaxLoss(output, target: target)
      loss.backward(to: [vocabVec, seqVec])
      adamOptimizer.step()
      return output
    }
    computeStream.joined()
    var correct = 0
    for k in 0..<deviceCount {
      let oneHot = oneHotGPU[k].toCPU()
      let output = DynamicGraph.Tensor<Float32>(output[k]).toCPU()
      for i in 0..<batchSize {
        let truth = oneHot[i, 1] > oneHot[i, 0]
        let prediction = output[i, 1] > output[i, 0]
        if truth == prediction {
          correct += 1
        }
      }
    }
    let accuracy = Double(correct) / Double(batchSize * deviceCount)
    overallAccuracy = overallAccuracy * 0.9 + accuracy * 0.1
    if adamOptimizer.step % 50 == 0 {
      print("epoch \(epoch) (\(i)/\(batchedTrainData.count)), training accuracy \(overallAccuracy)")
    }
  }
  graph.openStore("/home/liu/workspace/s4nnc/imdb.checkpoint") { store in
    store.write("vocab", variable: vocabVec)
    store.write("seq", variable: seqVec)
    store.write("transformer", model: transformer)
  }
}

// let batchedTestData = testData["tensor", "mask", "oneHot"].combine(size: batchSize)
// atchedTestData["squaredMask"] = batchedTestData["mask"].toOneSquared(maxLength: maxLength, variableLength: false)
