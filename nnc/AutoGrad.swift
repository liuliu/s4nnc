import C_nnc

public extension DynamicGraph.AnyTensor {
  func backward() {
  }
  func backward<S: Sequence>(to tensors: S) where S.Element == DynamicGraph.AnyTensor {
  }
}
