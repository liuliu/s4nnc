import nnc

let dynamicGraph = DynamicGraph()

let t0 = Tensor<Float32>(.CPU, .NHWC(1, 1, 1, 1))
t0[0, 0, 0, 0] = 1.23
let tv0 = dynamicGraph.variable(t0)
let t1 = Tensor<Float32>(.CPU, .NHWC(1, 1, 1, 1))
t1[0, 0, 0, 0] = 2
let tv1 = dynamicGraph.variable(t1)
let mul = Mul()
print(mul([tv0, tv1]))
