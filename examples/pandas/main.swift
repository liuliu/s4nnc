import nnc
import PythonKit

let graph = DynamicGraph()

let a: DynamicGraph.Tensor<Float> = graph.variable(.CPU, .C(1))
let b: DynamicGraph.Tensor<Float> = graph.variable(.CPU, .C(1))

a[0] = 1.2
b[0] = 2.2
let c = 2.2 * a .* (b * 3.3)
print(c[0])
