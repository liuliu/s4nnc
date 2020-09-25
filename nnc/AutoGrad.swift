import C_nnc

public extension DynamicGraph.AnyTensor {
  func backward() {
    var tensors = [DynamicGraph.AnyTensor]()
    // Not functioning.
    for v in self.graph.activeVariables {
      guard let tensor = v.tensor else { continue }
      guard tensor != self else { continue }
      tensors.append(tensor)
    }
    backward(to: tensors)
  }
  func backward<S: Sequence>(to tensors: S, streamContext: StreamContext? = nil) where S.Element: DynamicGraph.AnyTensor {
    let _graph = graph._graph
    let _inputs: [ccv_nnc_tensor_variable_t?] = tensors.map { $0._tensor }
    let inputSize = Int32(_inputs.count)
    let _outputs = UnsafeMutablePointer<ccv_nnc_tensor_variable_t?>.allocate(capacity: _inputs.count)
    for (i, tensor) in tensors.enumerated() {
      if tensor.grad == nil {
        tensor.grad = graph.variable()
      }
      (_outputs + i).initialize(to: tensor.grad!._tensor)
    }
    let _streamContext = streamContext?._stream
    var f: ccv_nnc_tensor_variable_t? = self._tensor
    ccv_nnc_dynamic_graph_backward(_graph, &f, 1, nil, _inputs, inputSize, _outputs, inputSize, _streamContext)
    _outputs.deallocate()
  }
}

public extension Sequence where Element: DynamicGraph.AnyTensor {
  func backward() {
  }
  func backward<S: Sequence>(to tensors: S) where S.Element: DynamicGraph.AnyTensor {
  }

}
