import C_nnc

public protocol Optimizer {
  var parameters: [DynamicGraph.AnyTensor] { get set }
  func step()
}
