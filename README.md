# Swift for NNC

s4nnc is a Swift interface for [libnnc](https://libnnc.org) library.

From the very start, [libnnc](https://libnnc.org) is meant to be a common runtime that supports many language bindings. It becomes apparent during the development that for deep learning, a raw C interface would be unwieldy complex to use in real-life. For example, the training loop of a transformer model for sentiment analysis takes more than 400 lines of code: <https://github.com/liuliu/ccv/blob/unstable/bin/nnc/imdb.c#L268>

A high-level language that delegates most of the work to the C runtime seems to be a win in brevity and usefulness. The same training loop of a transformer model in Swift takes less than 100 lines: <https://github.com/liuliu/s4nnc/blob/main/examples/imdb/main.swift#L197>

Because the heavy-lifting is done in the [libnnc](https://libnnc.org) library, the Swift portion can be light and even automatically generated. At the moment, we have about 3,000 lines of Swift code to run quite a few models on GPU, complete with data feeders, model specifications and optimizers.

## Runtime API

Currently, s4nnc works better under Linux with CUDA 11, CuDNN and NCCL. The API for s4nnc wraps around [Level-4](https://libnnc.org/api/level-4/) and [Level-5](https://libnnc.org/api/level-5/) C APIs.

### Tensor

```swift
public struct Tensor<Element> {
  init(_ kind: DeviceKind, _ dimensionFormat: TensorDimensionFormat)
  init<S: Sequence>(_ sequence: S, _ dimensionFormat: TensorDimensionFormat) where S.Element == Element
}
```

This method initialize a raw tensor that resides either on CPU or GPU with a given dimensions. Alternatively, you can initialize a tensor from native Swift array. Basic usage looks like this:

```swift
var tensor = Tensor<Float>(.CPU, .HWC(1, 1, 2))
tensor[0, 0, 0] = 1
tensor[0, 0, 1] = 2
```

There are very limited functionalities associated with raw tensors. Mostly, you can only `reshape` or `toGPU` / `toCPU`.

### DynamicGraph

`DynamicGraph` is where you associate most computations with tensors. The `DynamicGraph` operates on tensor variables / constants, not the raw tensors. Initializing a tensor variable / constant is very similar to initializing a raw tensor:

```swift
let graph = DynamicGraph()
let variable: DynamicGraph.Tensor<Float> = graph.variable(.CPU, .HWC(1, 1, 2))
```

A tensor variable can participate computations, for example:

```swift
let x: DynamicGraph.Tensor<Float> = graph.variable(.CPU, .C(1))
let y: DynamicGraph.Tensor<Float> = graph.variable(.CPU, .C(1))
x[0] = 2
y[0] = -1
let z = x .* y
print(z[0])
```

Because these are tensor variables, you can also do automatic differentiation:

```swift
x.requiresGrad = true
z.backward(to: [x])
print(DynamicGraph.Tensor<Float>(x.grad!)[0])
```

`requiresGrad` in above code merely denotes we need to populate the `grad` property of `x`. It doesn't carry other significance unlike in PyTorch.

Tensor variables memory management is automatic. If there is no reference to it (as defined by no automatic differentiation requires the given tensor variable's participation), the memory will be freed. Hence, unlike PyTorch, you don't need to worry about `no_grad` annotation most of the time.

### Model and Optimizer

Computations on `DynamicGraph` with tensor variables are stateless. s4nnc also provided stateful `Model` that contains trainable parameters. You can use `Model` to construct complex computation unit and train them.

```swift
func TwoLayerLinearModel() {
  let x = Input()
  let y = Dense(count: 2)(x)
  let z = Dense(count: 1)(y)
  return Model([x], [z])
}

let twoLayerModel = TwoLayerLinearModel()
let z = twoLayerModel(inputs: x)
print(z)
```

You can train the model with optimizers.

```swift
let sgd = SGDOptimizer(graph, nesterov: false, rate: 0.0001, scale: 1, decay: 0, momentum: 0.9, dampening: 0)
sgd.parameters = [twoLayerModel.parameters]
for _ 0..<100 {
  let x: DynamicGraph.Tensor<Float> = graph.variable(.CPU, .C(1))
  x[0] = 1
  let target: DynamicGraph.Tensor<Float> = graph.variable(.CPU, .C(1))
  target[0] = 0
  let z = twoLayerModel(inputs: x)
  let binaryLoss = SigmoidBinaryCrossEntropyLoss()
  let loss = binaryLoss(z, target: target)
  loss[0].backward(to: [x])
  sgd.step()
}
```

Because `Model` can express complex computations statically, it is recommended to have most of your computations expressed as `Model`.

### ModelBuilder

Sometimes, your `Model` can change its shape based on the inputs. `ModelBuilder` can take the input, and generate appropriate model. However, these models need to match on parameters. For example, if you have different length of text input to your transformer model, `ModelBuilder` can be helpful.

### DataFrame

`DataFrame` provides an easy way to construct data feeder into your computation. The data feeder is memory and computation efficient, meaning for each column you try to pull, only that column will be materialized. Hence, if you loop through a list of file names and declare a column to be the loaded images, only one image loaded at a time when you loop through the `DataFrame`.

```swift
let df = DataFrame(from: [filename1, filename2, filename3])
df["image"] = df["0"]!.toLoadImage()
for tensor in df["image", Tensor<UInt8>.self] {
  print(tensor)
}
```

We only load one image at a time, and the previous image is freed as soon as the next image pulled in.

The `DataFrame` object also provided basic functionalities to load from a CSV file. The CSV reader is considered to be fastest multi-core reader at the moment.

### StreamContext

Unlike PyTorch, s4nnc doesn't associate with implicit asynchronous stream when execute on GPU. To leverage asynchronous stream to improve computation efficiency, you can associate a `StreamContext` explicitly.

```swift
let computeStream = StreamContext(.GPU(0))
var z: DynamicGraph.Tensor<Float>? = nil
graph.withStream(computeStream) {
  let x: DynamicGraph.Tensor<Float> = graph.variable(.CPU, .C(1))
  let y: DynamicGraph.Tensor<Float> = graph.variable(.CPU, .C(1))
  x[0] = 2
  y[0] = -1
  z = x .+ y
}
computeStream.joined()
print(z[0]) // Result only available after joined the computeStream
```

### Storing Models and Tensors

A simple SQLite based data storage is provided from s4nnc. It is a key-value based storage for tensors, tensor variables and models. You can:

```swift
graph.openStore("filePath") { store in
  let aTensor = store.read("a")
  store.write("b", variable: z)
  store.write("2layer", model: twoLayerModel)
}
```

### Group

Multiple tensor variables can be grouped together for computations.

```swift
let xGroup = Group(x0, x1)
let yGroup = Group(y0, y1)
let zGroup = xGroup .* yGroup
```

This is useful because if tensor variables are on different GPUs, this can compute simultaneously. With `Model` and `Optimizer`, it is a transparent way to apply data parallelism to speed up your training loop.

## Example

Below are the training loop to train an sentiment analysis transformer model with s4nnc. It trains the model with multiple GPUs. You can find comparable PyTorch code from [Transformers from Scratch](http://peterbloem.nl/blog/transformers). You can find the rest of the code in <https://github.com/liuliu/s4nnc/blob/main/examples/imdb/main.swift>.

### Setup the Data Feeder Pipeline
```swift
var trainData = dataFromDisk(filePath: trainListFile)
// Extract tensors from ImdbText struct.
trainData["tensor"] = trainData["main", ImdbText.self].map(\.tensor)
trainData["mask"] = trainData["main", ImdbText.self].map(\.mask)
trainData["c"] = trainData["main", ImdbText.self].map(\.c)
// Create one hot tensor out of the scalar.
trainData["oneHot"] = trainData["c", Int.self].toOneHot(Float32.self, count: 2)

let deviceCount = DeviceKind.GPUInfo.count

// Batching tensors together. 
var batchedTrainData = trainData["tensor", "mask", "oneHot"].combine(size: batchSize, repeating: deviceCount)
for i in 0..<deviceCount {
  batchedTrainData["truncTensor_\(i)"] = batchedTrainData["tensor_\(i)"]!.toTruncate(batchedTrainData["mask_\(i)"]!)
  batchedTrainData["squaredMask_\(i)"] = batchedTrainData["mask_\(i)"]!.toOneSquared(maxLength: maxLength)
  // Move the tensors from CPU to GPU.
  let toGPUTrain = batchedTrainData["truncTensor_\(i)", "oneHot_\(i)", "squaredMask_\(i)"].toGPU(i)
  batchedTrainData["tensorGPU_\(i)"] = toGPUTrain["truncTensor_\(i)"]
  batchedTrainData["oneHotGPU_\(i)"] = toGPUTrain["oneHot_\(i)"]
  batchedTrainData["squaredMaskGPU_\(i)"] = toGPUTrain["squaredMask_\(i)"]
}
```

### The Training Loop
```swift
let graph = DynamicGraph()

let vocabVec: Group<DynamicGraph.Tensor<Float32>> = Group((0..<deviceCount).map { graph.variable(.GPU($0), .NC(vocabSize, embeddingSize)) })
let seqVec: Group<DynamicGraph.Tensor<Float32>> = Group((0..<deviceCount).map { graph.variable(.GPU($0), .NC(maxLength, embeddingSize)) })
vocabVec.rand(-1, 1)
seqVec.rand(-1, 1)
var adamOptimizer = AdamOptimizer(graph, step: 0, rate: 0.0001, beta1: 0.9, beta2: 0.98, decay: 0, epsilon: 1e-9)
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
    adamOptimizer.step = epoch * batchedTrainData.count + i + 1
    adamOptimizer.rate = 0.0001 * min(Float(adamOptimizer.step - 1) / (10000.0 / Float(batchSize)), 1) * Float(deviceCount)
    let tensorGPU = (0..<deviceCount).map { batch[$0 * 3] as! Tensor<Int32> }
    let oneHotGPU = (0..<deviceCount).map { batch[$0 * 3 + 1] as! Tensor<Float32> }
    let squaredMaskGPU = (0..<deviceCount).map { batch[$0 * 3 + 2] as! Tensor<Int32> }
    let batchLength = tensorGPU[0].dimensions[1]
    let output = graph.withStream(computeStream) { () -> Group<DynamicGraph.AnyTensor> in
      let wordIndices = graph.variable(tensorGPU.reshape(.C(batchSize * batchLength)))
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
      let inputVec = selectVec.reshape(.CHW(batchSize, batchLength, embeddingSize))
      let masked = graph.constant(squaredMaskGPU.reshape(.CHW(batchSize, batchLength, batchLength)))
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
    if adamOptimizer.step % 50  == 0 {
      print("epoch \(epoch) (\(i)/\(batchedTrainData.count)), training accuracy \(overallAccuracy)")
    }
  }
}
```
