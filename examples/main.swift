import nnc

let dynamicGraph = DynamicGraph()

let t0 = Tensor<Float32>(.CPU, .NHWC(1, 1, 1, 1))
t0[0, 0, 0, 0] = 1.23
let tv0 = dynamicGraph.variable(t0)
let t1 = Tensor<Float32>(.CPU, .NHWC(1, 1, 1, 1))
t1[0, 0, 0, 0] = 2
let tv1 = dynamicGraph.variable(t1)
let mul = Mul()
let tv2 = mul(tv0, tv1)
let t3 = Tensor<Float32>(.CPU, .NHWC(1, 1, 1, 1))
t3[0, 0, 0, 0] = -1
let tv3 = dynamicGraph.variable(t3)
let add = Add()
let tv4 = add(tv2, tv3)

print(tv4.rawValue[0, 0, 0, 0])
