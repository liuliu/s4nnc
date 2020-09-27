import nnc

let dynamicGraph = DynamicGraph()

var t0 = Tensor<Float32>(.CPU, .NHWC(1, 1, 1, 1))
t0[0, 0, 0, 0] = 1.23
let tv0 = dynamicGraph.variable(t0)
print(tv0)
var t1 = Tensor<Float32>(.CPU, .NHWC(1, 1, 1, 1))
t1[0, 0, 0, 0] = 2
let tv1 = dynamicGraph.variable(t1)
let tv2 = tv0 .* tv1
var t3 = Tensor<Float32>(.CPU, .NHWC(1, 1, 1, 1))
t3[0, 0, 0, 0] = -1
let tv3 = dynamicGraph.variable(t3)

let tv4 = tv2 + tv3
print(tv4.rawValue[0, 0, 0, 0])

[tv4].backward(to: [tv0])
tv4.backward(to: [tv0])
print(DynamicGraph.Tensor<Float32>(tv0.grad!).rawValue[0, 0, 0, 0])

func MulAdd() -> Model {
  let i0 = Input()
  let i1 = Input()
  let i2 = i0 .* i1
  let i3 = Input()
  let i4 = i2 - i3
  return Model([i0, i1, i3], [i4])
}

let muladd = MulAdd()
let tv5 = DynamicGraph.Tensor<Float32>(muladd([tv0, tv1, tv3])[0])
var tv5a = tv5.rawValue
tv5a[0, 0, 0, 0] = 10
print(tv5.rawValue[0, 0, 0, 0])
print(tv5a[0, 0, 0, 0])

let tv6 = dynamicGraph.variable(Tensor<Float32>([2, 3], .NC(2, 1)))
let tv7 = dynamicGraph.variable(Tensor<Float32>([3, 2], .NC(1, 2)))
let tv8 = tv6 * tv7
print(tv8.rawValue)
print(tv8.rawValue[0, 0])
print(tv8.rawValue[0, 1])
print(tv8.rawValue[1, 0])
print(tv8.rawValue[1, 1])
