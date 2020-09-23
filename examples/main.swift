import nnc

let dynamicGraph = DynamicGraph()

let t0 = Tensor<Float32>(.CPU, .NHWC(1, 1, 1, 1))
t0[0, 0, 0, 0] = 1.23
let tv0 = dynamicGraph.variable(t0)
let t1 = Tensor<Float32>(.CPU, .NHWC(1, 1, 1, 1))
t1[0, 0, 0, 0] = 2
let tv1 = dynamicGraph.variable(t1)
let tv2 = tv0 .* tv1
let t3 = Tensor<Float32>(.CPU, .NHWC(1, 1, 1, 1))
t3[0, 0, 0, 0] = -1
let tv3 = dynamicGraph.variable(t3)
let add = Add()
let tv4 = add(tv2, tv3)

print(tv4.rawValue[0, 0, 0, 0])

func MulAdd() -> Model {
  let i0 = Input()
  let i1 = Input()
  let i2 = i0 .* i1
  let i3 = Input()
  let i4 = i2 + i3
  return Model([i0, i1, i3], [i4])
}

let muladd = MulAdd()
let tv5 = DynamicGraph.Tensor<Float32>(muladd([tv0, tv1, tv3])[0])

print(tv5.rawValue[0, 0, 0, 0])
