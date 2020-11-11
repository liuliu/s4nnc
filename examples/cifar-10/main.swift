import nnc

let graph = DynamicGraph()

let tv1: DynamicGraph.Tensor<Float32> = graph.variable(.CPU, .C(1))
tv1[0] = 10
let tv1g0 = tv1.toGPU(0)
let tv1g1 = tv1.toGPU(1)
let tv2: DynamicGraph.Tensor<Float32> = graph.variable(.CPU, .C(1))
tv2[0] = 2
let tv2g0 = tv2.toGPU(0)
let tv2g1 = tv2.toGPU(1)

let model = Dense(count: 2)

let tv1g = [tv1g0, tv1g1]
let tv2g = [tv2g0, tv2g1]

let rv = model([tv1g])
print(rv)

/*
let tv3c0 = rv[0].toCPU()
let tv3c1 = rv[1].toCPU()
print(rv)
*/
