import nnc
import Foundation

/**
 * MARK - Setup the Transformer Model
 */

func SelfAttention(k: Int, h: Int, b: Int, t: Int, dropout: Float) -> Model
{
  let x = Input()
  let mask = Input()
  let multiheads = x.reshape([b * t, k])
  let tokeys = Dense(count: k * h, noBias: true)
  let toqueries = Dense(count: k * h, noBias: true)
  let tovalues = Dense(count: k * h, noBias: true)
  let keys = tokeys(multiheads).reshape([t, b, h, k]).transpose(0, 2).reshape([b * h, t, k])
  let queries = toqueries(multiheads).reshape([t, b, h, k]).transpose(0, 2).reshape([b * h, t, k])
  let values = tovalues(multiheads).reshape([t, b, h, k]).transpose(0, 2).reshape([b * h, t, k])
  var dot = Matmul(transposeB: (1, 2))(queries, keys)
  dot = (1.0 / Float(k).squareRoot()) * dot
  dot = MaskedFill(equalTo: 0, fillWith: 1e-9)(dot, mask)
  dot = dot.reshape([b * h * t, t])
  dot = Softmax()(dot)
  if dropout > 0 {
    dot = Dropout(probability: dropout)(dot)
  }
  dot = dot.reshape([b * h, t, t])
  var out = dot * values
  out = out.reshape([h, b, t, k]).transpose(0, 2).reshape([b * t, h * k])
  let unifyheads = Dense(count: k)
  out = unifyheads(out).reshape([t, b, k])
  return Model([x, mask], [out])
}

func TransformerBlock(k: Int, h: Int, b: Int, t: Int, ff: Int, dropout: Float) -> Model
{
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
  out = out.reshape([b * t, k])
  out = Dense(count: ff)(out)
  out = RELU()(out)
  out = Dense(count: k)(out)
  out = out.reshape([t, b, k])
  out = first + out
  out = LayerNorm(epsilon: 1e-5, axis: [2])(out)
  if dropout > 0 {
    out = Dropout(probability: dropout)(out)
  }
  return Model([x, mask], [out])
}

func ClassicTransformer(layers: Int, k: Int, h: Int, b: Int, t: Int, ff: Int, dropout: Float) -> Model
{
  let x = Input()
  let mask = Input()
  var out = x.transpose(0, 1)
  for _ in 0..<layers {
    out = TransformerBlock(k: k, h: h, b: b, t: t, ff: ff, dropout: dropout)(out, mask)
  }
  out = out.transpose(0, 1).transpose(1, 2).reshape([b, k, t, 1])
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

/**
 * MARK - The Training Program
 */

let parameters = TransformerParameter(ff: 4, layers: 2, h: 8, dropout: 0.1)

let transformer: ModelBuilder = ModelBuilder { inputs in
  let b = inputs[0].dimensions[0]
  let t = inputs[0].dimensions[1]
  let k = inputs[0].dimensions[2]
  return ClassicTransformer(layers: parameters.layers, k: k, h: parameters.h, b: b, t: t, ff: parameters.ff * k, dropout: parameters.dropout)
}

let graph = DynamicGraph()

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
/**
 * MARK - Data Processing
 */

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
    let separators: Set<Character> = [" ", ".", ",", "<", ">", "/", "~", "`", "@", "#", "$", "%", "^", "&", "*", "+", "\\", "\""]
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

/**
 * MARK - Setup the Data Feeder Pipeline
 */

let trainData = dataFromDisk(filePath: trainListFile)
let testData = dataFromDisk(filePath: testListFile)
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

// Batching tensors together. 
let batchedTrainData = DataFrame(batchOf: trainData["tensor", "mask", "oneHot"], size: batchSize)
batchedTrainData["truncTensor"] = batchedTrainData["tensor"]!.toTruncate(batchedTrainData["mask"]!)
batchedTrainData["squaredMask"] = batchedTrainData["mask"]!.toOneSquared(maxLength: maxLength)
// Move the tensors from CPU to GPU.
let toGPUTrain = batchedTrainData["truncTensor", "oneHot", "squaredMask"].toGPU()
batchedTrainData["tensorGPU"] = toGPUTrain["truncTensor"]
batchedTrainData["oneHotGPU"] = toGPUTrain["oneHot"]
batchedTrainData["squaredMaskGPU"] = toGPUTrain["squaredMask"]

/**
 * The Training Loop
 */
let vocabVec: DynamicGraph.Tensor<Float32> = graph.variable(.GPU(0), .NC(vocabSize, embeddingSize))
let seqVec: DynamicGraph.Tensor<Float32> = graph.variable(.GPU(0), .NC(maxLength, embeddingSize))
vocabVec.rand(-1, 1)
seqVec.rand(-1, 1)
var adamOptimizer = AdamOptimizer(graph, step: 0, rate: 0.0001, beta1: 0.9, beta2: 0.98, decay: 0, epsilon: 1e-9)
adamOptimizer.parameters = [vocabVec, seqVec]
var overallAccuracy = 0.0
for epoch in 0..<10 {
  batchedTrainData.shuffle()
  for (i, batch) in batchedTrainData["tensorGPU", "oneHotGPU", "squaredMaskGPU"].enumerated() {
    adamOptimizer.step = epoch * batchedTrainData.count + i + 1
    adamOptimizer.rate = 0.0001 * min(Float(i) / (10000.0 / Float(batchSize)), 1)
    let tensorGPU = batch[0] as! Tensor<Int32>
    let oneHotGPU = batch[1] as! Tensor<Float32>
    let squaredMaskGPU = batch[2] as! Tensor<Int32>
    let batchLength = tensorGPU.dimensions[1]
    let wordIndices = graph.variable(tensorGPU.reshape(.C(batchSize * batchLength)))
    let wordVec = Functional.indexSelect(input: vocabVec, index: wordIndices)
    var seqIndicesCPU = Tensor<Int32>(.CPU, .C(batchSize * batchLength))
    for i in 0..<batchSize {
      for j in 0..<batchLength {
        seqIndicesCPU[i * batchLength + j] = Int32(j)
      }
    }
    let seqIndicesGPU = seqIndicesCPU.toGPU()
    let seqIndices = graph.constant(seqIndicesGPU)
    let posVec = Functional.indexSelect(input: seqVec, index: seqIndices)
    let selectVec = wordVec + posVec
    let inputVec = selectVec.reshape(.CHW(batchSize, batchLength, embeddingSize))
    let masked = graph.constant(squaredMaskGPU.reshape(.CHW(batchSize, batchLength, batchLength)))
    let outputs = transformer([inputVec, masked])
    let softmaxLoss = SoftmaxCrossEntropyLoss()
    let target = graph.variable(oneHotGPU)
    let loss = softmaxLoss(outputs[0], target: target)
    loss.backward(to: [vocabVec, seqVec])
    if (i + 1) % 2 == 0 {
      adamOptimizer.step()
    }
    let oneHot = oneHotGPU.toCPU()
    let output = DynamicGraph.Tensor<Float32>(outputs[0]).toCPU()
    var correct = 0
    for i in 0..<batchSize {
      let truth = oneHot[i, 1] > oneHot[i, 0]
      let prediction = output[i, 1] > output[i, 0]
      if truth == prediction {
        correct += 1
      }
    }
    let accuracy = Double(correct) / Double(batchSize)
    overallAccuracy = overallAccuracy * 0.975 + accuracy * 0.025
    if adamOptimizer.step % 50  == 0 {
      print("epoch \(epoch) (\(i)/\(batchedTrainData.count)), training accuracy \(overallAccuracy)")
    }
  }
}

// let batchedTestData = DataFrame(batchOf: testData["tensor", "mask", "oneHot"], size: batchSize)
// atchedTestData["squaredMask"] = batchedTestData["mask"].toOneSquared(maxLength: maxLength, variableLength: false)
