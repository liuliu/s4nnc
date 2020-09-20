import nnc

let dynamicGraph = DynamicGraph()

let tensor: DynamicGraph.Tensor<Float64> = dynamicGraph.variable(.CPU, .NHWC(1, 1, 1, 1))
print(tensor)
