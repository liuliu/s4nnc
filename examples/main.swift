import nnc

let graph = DynamicGraph()

let a: DynamicGraph.Tensor<Float> = graph.variable(.CPU, .C(1))
let b: DynamicGraph.Tensor<Float> = graph.variable(.CPU, .C(1))

a[0] = 1.2
b[0] = 2.2

let gpu_a = a.toGPU()
let gpu_b = b.toGPU()
let c = gpu_a .* gpu_b
let cpu_c = c.toCPU()
print(cpu_c[0])
