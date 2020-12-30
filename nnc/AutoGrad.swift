import C_nnc

extension DynamicGraph.AnyTensor {
  /**
   * Compute gradients from this tensor to the source tensors.
   *
   * - Parameters:
   *   - to: The source tensors to this tensor.
   *   - streamContext: The stream context to perform such operation.
   */
  public func backward<S: Sequence>(to tensors: S, streamContext: StreamContext? = nil)
  where S.Element: DynamicGraph.AnyTensor {
    let _graph = graph._graph
    var gradients = graph.gradients(for: [self])
    var gradientsSet = Set(gradients)
    for tensor in tensors {
      if !gradientsSet.contains(tensor) {
        gradients.append(tensor)
        gradientsSet.insert(tensor)
      }
    }
    let _inputs: [ccv_nnc_tensor_variable_t?] = gradients.map { $0._tensor }
    let inputSize = Int32(_inputs.count)
    let _outputs = UnsafeMutablePointer<ccv_nnc_tensor_variable_t?>.allocate(
      capacity: _inputs.count)
    for (i, tensor) in gradients.enumerated() {
      precondition(!tensor.isConstant)
      if tensor.grad == nil && tensor.requiresGrad {
        tensor.grad = graph.variable()
      }
      (_outputs + i).initialize(to: tensor.grad?._tensor)
    }
    let _streamContext = streamContext?._stream
    var f: ccv_nnc_tensor_variable_t? = self._tensor
    var g: ccv_nnc_tensor_variable_t? = self.grad?._tensor
    if g == nil {
      ccv_nnc_dynamic_graph_backward(
        _graph, &f, 1, nil, _inputs, inputSize, _outputs, inputSize, _streamContext)
    } else {
      ccv_nnc_dynamic_graph_backward(
        _graph, &f, 1, &g, _inputs, inputSize, _outputs, inputSize, _streamContext)
    }
    _outputs.deallocate()
  }

  /**
   * Compute gradients from this tensor to the source tensor.
   *
   * - Parameters:
   *   - to: The source tensor to this tensor.
   *   - streamContext: The stream context to perform such operation.
   */
  public func backward(to tensor: DynamicGraph.AnyTensor, streamContext: StreamContext? = nil) {
    backward(to: [tensor], streamContext: streamContext)
  }
}

extension DynamicGraph.Group {
  /**
   * Compute gradients from this tensor to the source tensors.
   *
   * - Parameters:
   *   - to: The source tensors to this tensor.
   *   - streamContext: The stream context to perform such operation.
   */
  public func backward<S: Sequence>(to tensors: S, streamContext: StreamContext? = nil)
  where S.Element: DynamicGraph.AnyGroup {
    precondition(underlyingArray.count > 0)
    let graph = underlyingArray[0].graph
    var gradients = graph.gradients(for: underlyingArray)
    var gradientsSet = Set(gradients)
    for tensor in tensors.flatMap({ $0.underlying }) {
      if !gradientsSet.contains(tensor) {
        gradients.append(tensor)
        gradientsSet.insert(tensor)
      }
    }
    let _graph = graph._graph
    let _inputs: [ccv_nnc_tensor_variable_t?] = gradients.map { $0._tensor }
    let inputSize = Int32(_inputs.count)
    let _outputs = UnsafeMutablePointer<ccv_nnc_tensor_variable_t?>.allocate(
      capacity: _inputs.count)
    for (i, tensor) in gradients.enumerated() {
      precondition(!tensor.isConstant)
      if tensor.grad == nil && tensor.requiresGrad {
        tensor.grad = graph.variable()
      }
      (_outputs + i).initialize(to: tensor.grad?._tensor)
    }
    let _streamContext = streamContext?._stream
    let f: [ccv_nnc_tensor_variable_t?] = self.underlyingArray.map { $0._tensor }
    let g: [ccv_nnc_tensor_variable_t?] = self.underlyingArray.map { $0.grad?._tensor }
    ccv_nnc_dynamic_graph_backward(
      _graph, f, Int32(f.count), g, _inputs, inputSize, _outputs, inputSize, _streamContext)
    _outputs.deallocate()
  }

  /**
   * Compute gradients from this tensor to the source tensor.
   *
   * - Parameters:
   *   - to: The source tensor to this tensor.
   *   - streamContext: The stream context to perform such operation.
   */
  public func backward<Group: DynamicGraph.AnyGroup>(
    to tensor: Group, streamContext: StreamContext? = nil
  ) {
    backward(to: [tensor], streamContext: streamContext)
  }
}

extension Collection where Element: DynamicGraph.AnyTensor {
  /**
   * Compute gradients from this tensor to the source tensors.
   *
   * - Parameters:
   *   - to: The source tensors to this tensor.
   *   - streamContext: The stream context to perform such operation.
   */
  public func backward<S: Sequence>(to tensors: S, streamContext: StreamContext? = nil)
  where S.Element: DynamicGraph.AnyTensor {
    precondition(self.count > 0)
    let graph = self.first!.graph
    for f in self {
      assert(f.graph === graph)
    }
    let _graph = graph._graph
    var gradients = graph.gradients(for: self)
    var gradientsSet = Set(gradients)
    for tensor in tensors {
      if !gradientsSet.contains(tensor) {
        gradients.append(tensor)
        gradientsSet.insert(tensor)
      }
    }
    let _inputs: [ccv_nnc_tensor_variable_t?] = gradients.map { $0._tensor }
    let inputSize = Int32(_inputs.count)
    let _outputs = UnsafeMutablePointer<ccv_nnc_tensor_variable_t?>.allocate(
      capacity: _inputs.count)
    for (i, tensor) in gradients.enumerated() {
      precondition(!tensor.isConstant)
      if tensor.grad == nil && tensor.requiresGrad {
        tensor.grad = graph.variable()
      }
      (_outputs + i).initialize(to: tensor.grad?._tensor)
    }
    let _streamContext = streamContext?._stream
    let f: [ccv_nnc_tensor_variable_t?] = self.map { $0._tensor }
    let g: [ccv_nnc_tensor_variable_t?] = self.map { $0.grad?._tensor }
    ccv_nnc_dynamic_graph_backward(
      _graph, f, Int32(f.count), g, _inputs, inputSize, _outputs, inputSize, _streamContext)
    _outputs.deallocate()
  }

  /**
   * Compute gradients from this tensor to the source tensor.
   *
   * - Parameters:
   *   - to: The source tensor to this tensor.
   *   - streamContext: The stream context to perform such operation.
   */
  public func backward(to tensor: DynamicGraph.AnyTensor, streamContext: StreamContext? = nil) {
    backward(to: [tensor], streamContext: streamContext)
  }
}

extension Collection where Element: DynamicGraph.AnyGroup {
  /**
   * Compute gradients from this tensor to the source tensors.
   *
   * - Parameters:
   *   - to: The source tensors to this tensor.
   *   - streamContext: The stream context to perform such operation.
   */
  public func backward<S: Sequence>(to tensors: S, streamContext: StreamContext? = nil)
  where S.Element: DynamicGraph.AnyGroup {
    precondition(self.count > 0)
    let graph = self.first!.underlying[0].graph
    for group in self {
      for f in group.underlying {
        assert(f.graph === graph)
      }
    }
    let _graph = graph._graph
    var gradients = graph.gradients(for: self.flatMap { $0.underlying })
    var gradientsSet = Set(gradients)
    for tensor in tensors.flatMap({ $0.underlying }) {
      if !gradientsSet.contains(tensor) {
        gradients.append(tensor)
        gradientsSet.insert(tensor)
      }
    }
    let _inputs: [ccv_nnc_tensor_variable_t?] = gradients.map { $0._tensor }
    let inputSize = Int32(_inputs.count)
    let _outputs = UnsafeMutablePointer<ccv_nnc_tensor_variable_t?>.allocate(
      capacity: _inputs.count)
    for (i, tensor) in gradients.enumerated() {
      precondition(!tensor.isConstant)
      if tensor.grad == nil && tensor.requiresGrad {
        tensor.grad = graph.variable()
      }
      (_outputs + i).initialize(to: tensor.grad?._tensor)
    }
    let _streamContext = streamContext?._stream
    let f: [ccv_nnc_tensor_variable_t?] = self.flatMap { $0.underlying.map { $0._tensor } }
    let g: [ccv_nnc_tensor_variable_t?] = self.flatMap { $0.underlying.map { $0.grad?._tensor } }
    ccv_nnc_dynamic_graph_backward(
      _graph, f, Int32(f.count), g, _inputs, inputSize, _outputs, inputSize, _streamContext)
    _outputs.deallocate()
  }

  /**
   * Compute gradients from this tensor to the source tensor.
   *
   * - Parameters:
   *   - to: The source tensor to this tensor.
   *   - streamContext: The stream context to perform such operation.
   */
  public func backward<Group: DynamicGraph.AnyGroup>(
    to tensor: Group, streamContext: StreamContext? = nil
  ) {
    backward(to: [tensor], streamContext: streamContext)
  }
}
