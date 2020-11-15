import C_nnc

public extension DynamicGraph.AnyTensor {
  func backward<S: Sequence>(to tensors: S, streamContext: StreamContext? = nil) where S.Element: DynamicGraph.AnyTensor {
    let _graph = graph._graph
    let _inputs: [ccv_nnc_tensor_variable_t?] = tensors.map { $0._tensor }
    let inputSize = Int32(_inputs.count)
    let _outputs = UnsafeMutablePointer<ccv_nnc_tensor_variable_t?>.allocate(capacity: _inputs.count)
    for (i, tensor) in tensors.enumerated() {
      if tensor.grad == nil && !tensor.isConstant {
        tensor.grad = graph.variable()
      }
      (_outputs + i).initialize(to: tensor.grad?._tensor)
    }
    let _streamContext = streamContext?._stream
    var f: ccv_nnc_tensor_variable_t? = self._tensor
    ccv_nnc_dynamic_graph_backward(_graph, &f, 1, nil, _inputs, inputSize, _outputs, inputSize, _streamContext)
    _outputs.deallocate()
  }

  func backward(to tensor: DynamicGraph.AnyTensor, streamContext: StreamContext? = nil) {
    backward(to: [tensor], streamContext: streamContext)
  }
}

public extension DynamicGraph.Group {
  func backward<S: Sequence>(to tensors: S, streamContext: StreamContext? = nil) where S.Element: DynamicGraph.AnyGroup {
    precondition(underlyingArray.count > 0)
    let graph = underlyingArray[0].graph
    let _graph = graph._graph
    let _inputs: [ccv_nnc_tensor_variable_t?] = tensors.flatMap { $0.underlying.map { $0._tensor } }
    let inputSize = Int32(_inputs.count)
    let _outputs = UnsafeMutablePointer<ccv_nnc_tensor_variable_t?>.allocate(capacity: _inputs.count)
    var i = 0
    for group in tensors {
      for tensor in group.underlying {
        if tensor.grad == nil && !tensor.isConstant {
          tensor.grad = graph.variable()
        }
        (_outputs + i).initialize(to: tensor.grad?._tensor)
        i += 1
      }
    }
    let _streamContext = streamContext?._stream
    let f: [ccv_nnc_tensor_variable_t?] = self.underlyingArray.map { $0._tensor }
    ccv_nnc_dynamic_graph_backward(_graph, f, Int32(f.count), nil, _inputs, inputSize, _outputs, inputSize, _streamContext)
    _outputs.deallocate()
  }

  func backward<Group: DynamicGraph.AnyGroup>(to tensor: Group, streamContext: StreamContext? = nil) {
    backward(to: [tensor], streamContext: streamContext)
  }
}

public extension Collection where Element: DynamicGraph.AnyTensor {
  func backward<S: Sequence>(to tensors: S, streamContext: StreamContext? = nil) where S.Element: DynamicGraph.AnyTensor {
    precondition(self.count > 0)
    let graph = self.first!.graph
    for f in self {
      assert(f.graph === graph)
    }
    let _graph = graph._graph
    let _inputs: [ccv_nnc_tensor_variable_t?] = tensors.map { $0._tensor }
    let inputSize = Int32(_inputs.count)
    let _outputs = UnsafeMutablePointer<ccv_nnc_tensor_variable_t?>.allocate(capacity: _inputs.count)
    for (i, tensor) in tensors.enumerated() {
      if tensor.grad == nil && !tensor.isConstant {
        tensor.grad = graph.variable()
      }
      (_outputs + i).initialize(to: tensor.grad?._tensor)
    }
    let _streamContext = streamContext?._stream
    let f: [ccv_nnc_tensor_variable_t?] = self.map { $0._tensor }
    ccv_nnc_dynamic_graph_backward(_graph, f, Int32(f.count), nil, _inputs, inputSize, _outputs, inputSize, _streamContext)
    _outputs.deallocate()
  }

  func backward(to tensor: DynamicGraph.AnyTensor, streamContext: StreamContext? = nil) {
    backward(to: [tensor], streamContext: streamContext)
  }
}

public extension Collection where Element: DynamicGraph.AnyGroup {
  func backward<S: Sequence>(to tensors: S, streamContext: StreamContext? = nil) where S.Element: DynamicGraph.AnyGroup {
    precondition(self.count > 0)
    let graph = self.first!.underlying[0].graph
    for group in self {
      for f in group.underlying {
        assert(f.graph === graph)
      }
    }
    let _graph = graph._graph
    let _inputs: [ccv_nnc_tensor_variable_t?] = tensors.flatMap { $0.underlying.map { $0._tensor } }
    let inputSize = Int32(_inputs.count)
    let _outputs = UnsafeMutablePointer<ccv_nnc_tensor_variable_t?>.allocate(capacity: _inputs.count)
    var i = 0
    for group in tensors {
      for tensor in group.underlying {
        if tensor.grad == nil && !tensor.isConstant {
          tensor.grad = graph.variable()
        }
        (_outputs + i).initialize(to: tensor.grad?._tensor)
        i += 1
      }
    }
    let _streamContext = streamContext?._stream
    let f: [ccv_nnc_tensor_variable_t?] = self.flatMap { $0.underlying.map { $0._tensor } }
    ccv_nnc_dynamic_graph_backward(_graph, f, Int32(f.count), nil, _inputs, inputSize, _outputs, inputSize, _streamContext)
    _outputs.deallocate()
  }

  func backward<Group: DynamicGraph.AnyGroup>(to tensor: Group, streamContext: StreamContext? = nil) {
    backward(to: [tensor], streamContext: streamContext)
  }
}
