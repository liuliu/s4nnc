import nnc

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

