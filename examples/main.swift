import nnc
import PythonKit

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
  out = Dense(count: 1)(out)
  return Model([x, mask], [out])
}

struct TransformerParameter {
  var ff: Int
  var layers: Int
  var h: Int
  var dropout: Float
}

let parameters = TransformerParameter(ff: 8, layers: 6, h: 10, dropout: 0.5)

let dynamicClassicTransformer: ModelBuilder = ModelBuilder { inputs in
  let b = inputs[0].dimensions[0]
  let t = inputs[0].dimensions[1]
  let k = inputs[0].dimensions[2]
  return ClassicTransformer(layers: parameters.layers, k: k, h: parameters.h, b: b, t: t, ff: parameters.ff * k, dropout: parameters.dropout)
}

let graph = DynamicGraph()

let a: DynamicGraph.Tensor<Float> = graph.variable(.CPU, .C(1))
let b: DynamicGraph.Tensor<Float> = graph.variable(.CPU, .C(1))

a[0] = 1.2
b[0] = 2.2
let c = 2.2 * a .* (b * 3.3)
print(c[0])
