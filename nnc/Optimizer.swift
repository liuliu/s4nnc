import C_nnc

public protocol Optimizer {
  var parameters: [DynamicGraph.AnyTensor] { get set }
  func step(streamContext: StreamContext?)
}

public extension Optimizer {
  func step() {
    step(streamContext: nil)
  }
}

extension Optimizer {
  fileprivate func step(graph: DynamicGraph, minimizer: ccv_nnc_cmd_t, savedAux: [DynamicGraph.AnyTensor], streamContext: StreamContext?) {
    for parameter in parameters {
      assert(parameter.graph === graph)
    }
    for aux in savedAux {
      assert(aux.graph === graph)
    }
    let _graph = graph._graph
    let _streamContext = streamContext?._stream
    guard parameters.count > 0 else {
      ccv_nnc_dynamic_graph_apply_gradients(_graph, minimizer, nil, 0, nil, 0, nil, 0, _streamContext)
      return
    }
    let _gradients: [ccv_nnc_tensor_variable_t?] = parameters.map { $0.grad?._tensor }
    let _parameters = UnsafeMutablePointer<ccv_nnc_tensor_variable_t?>.allocate(capacity: parameters.count)
    for (i, variable) in parameters.enumerated() {
      (_parameters + i).initialize(to: variable._tensor)
    }
    let _savedAux = UnsafeMutablePointer<ccv_nnc_tensor_variable_t?>.allocate(capacity: savedAux.count)
    for (i, variable) in savedAux.enumerated() {
      (_savedAux + i).initialize(to: variable._tensor)
    }
    let parameterSize = Int32(parameters.count)
    ccv_nnc_dynamic_graph_apply_gradients(_graph, minimizer, _gradients, parameterSize, _parameters, parameterSize, _savedAux, 0, _streamContext)
    _parameters.deallocate()
    _savedAux.deallocate()
  }
}

public struct SGDOptimizer: Optimizer {
  private let graph: DynamicGraph
  public var nesterov: Bool
  public var rate: Float
  public var scale: Float
  public var decay: Float
  public var momentum: Float
  public var dampening: Float
  public var parameters = [DynamicGraph.AnyTensor]() {
    didSet {
      guard parameters.count > 0 else { return }
      let graph = parameters.first!.graph
      for parameter in parameters {
        assert(parameter.graph === graph)
      }
      // Update private saved_aux.
      let size = Int(ccv_nnc_minimizer_saved_aux_size(minimizer))
      savedAux = (0..<(parameters.count * size)).map { _ in graph.variable() }
    }
  }

  private var savedAux = [DynamicGraph.AnyTensor]()
  private var minimizer: ccv_nnc_cmd_t {
    var params = ccv_nnc_cmd_param_t()
    params.sgd.nesterov = nesterov ? 1 : 0
    params.sgd.rate = rate
    params.sgd.scale = scale
    params.sgd.decay = decay
    params.sgd.momentum = momentum
    params.sgd.dampening = dampening
    return ccv_nnc_cmd(CCV_NNC_SGD_FORWARD, nil, params, 0)
  }

  public init(_ graph: DynamicGraph, nesterov: Bool, rate: Float, scale: Float, decay: Float, momentum: Float, dampening: Float) {
    self.graph = graph
    self.nesterov = nesterov
    self.rate = rate
    self.scale = scale
    self.decay = decay
    self.momentum = momentum
    self.dampening = dampening
  }

  public func step(streamContext: StreamContext?) {
    step(graph: graph, minimizer: minimizer, savedAux: savedAux, streamContext: streamContext)
  }
}

public struct AdamOptimizer: Optimizer {
  private let graph: DynamicGraph
  public var step: Int
  public var rate: Float
  public var beta1: Float
  public var beta2: Float
  public var decay: Float
  public var epsilon: Float
  public var parameters = [DynamicGraph.AnyTensor]() {
    didSet {
      precondition(parameters.count > 0)
      let graph = parameters.first!.graph
      for parameter in parameters {
        assert(parameter.graph === graph)
      }
      // Update private saved_aux.
      let size = Int(ccv_nnc_minimizer_saved_aux_size(minimizer))
      savedAux = (0..<(parameters.count * size)).map { _ in graph.variable() }
    }
  }

  private var savedAux = [DynamicGraph.AnyTensor]()
  private var minimizer: ccv_nnc_cmd_t {
    var params = ccv_nnc_cmd_param_t()
    params.adam.step = Int32(step)
    params.adam.rate = rate
    params.adam.beta1 = beta1
    params.adam.beta2 = beta2
    params.adam.decay = decay
    params.adam.epsilon = epsilon
    return ccv_nnc_cmd(CCV_NNC_ADAM_FORWARD, nil, params, 0)
  }

  public init(_ graph: DynamicGraph, step: Int, rate: Float, beta1: Float, beta2: Float, decay: Float, epsilon: Float) {
    self.graph = graph
    self.step = step
    self.rate = rate
    self.beta1 = beta1
    self.beta2 = beta2
    self.decay = decay
    self.epsilon = epsilon
  }

  public func step(streamContext: StreamContext?) {
    step(graph: graph, minimizer: minimizer, savedAux: savedAux, streamContext: streamContext)
  }
}
