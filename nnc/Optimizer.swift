import C_nnc

public protocol Optimizer {
  var parameters: [DynamicGraph_AnyParameters] { get set }
  func step(streamContext: StreamContext?)
}

public extension Optimizer {
  func step() {
    step(streamContext: nil)
  }
}

fileprivate func _step(graph: DynamicGraph, minimizer: ccv_nnc_cmd_t, parameters: [DynamicGraph.AnyTensor], savedAux: [DynamicGraph.AnyTensor], streamContext: StreamContext?) {
  for parameter in parameters {
    assert(parameter.graph === graph)
  }
  for aux in savedAux {
    assert(aux.graph === graph)
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
  let _graph = graph._graph
  let _streamContext = (streamContext ?? graph.streamContext)?._stream
  ccv_nnc_dynamic_graph_apply_gradients(_graph, minimizer, _gradients, parameterSize, _parameters, parameterSize, _savedAux, 0, _streamContext)
  _parameters.deallocate()
  _savedAux.deallocate()
  for parameter in parameters {
    parameter.grad = nil
  }
}

fileprivate func _step(graph: DynamicGraph, minimizer: ccv_nnc_cmd_t, parameters: [DynamicGraph.AnyGroup], savedAux: [DynamicGraph.AnyGroup], streamContext: StreamContext?) {
  precondition(parameters.count > 0)
  let parallel = parameters[0].underlying.count
  precondition(parallel > 0)
  for group in parameters {
    for parameter in group.underlying {
      assert(parameter.graph === graph)
    }
  }
  for group in savedAux {
    for aux in group.underlying {
      assert(aux.graph === graph)
    }
  }
  let parameterSize = parameters.count
  var _gradients = [ccv_nnc_tensor_variable_t?](repeating: nil, count: parallel * parameterSize)
  let _parameters = UnsafeMutablePointer<ccv_nnc_tensor_variable_t?>.allocate(capacity: parameterSize * parallel)
  for (i, group) in parameters.enumerated() {
    for (j, variable) in group.underlying.enumerated() {
      _gradients[j * parameterSize + i] = variable.grad?._tensor
      (_parameters + j * parameterSize + i).initialize(to: variable._tensor)
    }
  }
  let savedAuxSize = savedAux.count
  let _savedAux = UnsafeMutablePointer<ccv_nnc_tensor_variable_t?>.allocate(capacity: savedAuxSize * parallel)
  for (i, group) in savedAux.enumerated() {
    for (j, variable) in group.underlying.enumerated() {
      (_savedAux + j * savedAuxSize + i).initialize(to: variable._tensor)
    }
  }
  let _graph = graph._graph
  let _streamContext = (streamContext ?? graph.streamContext)?._stream
  ccv_nnc_dynamic_graph_apply_gradients(_graph, minimizer, _gradients, Int32(parameterSize * parallel), _parameters, Int32(parameterSize * parallel), _savedAux, Int32(parallel), _streamContext)
  _parameters.deallocate()
  _savedAux.deallocate()
  for group in parameters {
    for parameter in group.underlying {
      parameter.grad = nil
    }
  }
}

func optimizerStep(graph: DynamicGraph, minimizer: ccv_nnc_cmd_t, parameters: [DynamicGraph_AnyParameters], savedAux: [DynamicGraph_Any], streamContext: StreamContext?) {
  let modelParameters = parameters.compactMap { $0 as? Model.Parameters }
  let primaryModelParameters = modelParameters.filter { $0._io == $0.model!._parameters && $0.model!.owner == nil }
  var models = Set(modelParameters.map { $0.model!.owner ?? $0.model! })
  // Reset these models with primary parameters with the new minimizer.
  for parameter in primaryModelParameters {
    let model = parameter.model!
    models.remove(model)
    ccv_cnnp_model_set_minimizer(model._model, minimizer, 1, nil, 0)
  }
  // Reset other models to use noop.
  let params = CmdParamsFactory.factory.newParams()
  let noop = ccv_nnc_cmd(CCV_NNC_NOOP, nil, params, 0)
  for model in models {
    ccv_cnnp_model_set_minimizer(model._model, noop, 1, nil, 0)
  }
  // Set minimizers on other models.
  var modelParametersMap = [Model: [Model.Parameters]]()
  for parameter in modelParameters
    // If parameter is not primary
    where parameter._io != parameter.model!._parameters || parameter.model!.owner != nil {
    let model = parameter.model!.owner ?? parameter.model!
    modelParametersMap[model, default: [Model.Parameters]()].append(parameter)
  }
  for (model, parameters) in modelParametersMap {
    let _parameters: [ccv_cnnp_model_io_t?] = parameters.map { $0._io }
    ccv_cnnp_model_set_minimizer(model._model, minimizer, 0, _parameters, Int32(_parameters.count))
  }
  let tensorParameters = parameters.compactMap { $0 as? DynamicGraph_Any }
  assert(modelParameters.count + tensorParameters.count == parameters.count)
  guard tensorParameters.count > 0 else {
    let _graph = graph._graph
    let _streamContext = (streamContext ?? graph.streamContext)?._stream
    ccv_nnc_dynamic_graph_apply_gradients(_graph, minimizer, nil, 0, nil, 0, nil, 0, _streamContext)
    return
  }
  switch tensorParameters[0] {
  case is DynamicGraph.AnyTensor:
    _step(graph: graph, minimizer: minimizer, parameters: tensorParameters as! [DynamicGraph.AnyTensor], savedAux: savedAux as! [DynamicGraph.AnyTensor], streamContext: streamContext)
  case is DynamicGraph.AnyGroup:
    _step(graph: graph, minimizer: minimizer, parameters: tensorParameters as! [DynamicGraph.AnyGroup], savedAux: savedAux as! [DynamicGraph.AnyGroup], streamContext: streamContext)
  default:
    fatalError("Cannot support the given type")
  }
}

extension Optimizer {

  func savedAux(minimizer: ccv_nnc_cmd_t) -> [DynamicGraph_Any] {
    let parameters = self.parameters.compactMap { $0 as? DynamicGraph_Any }
    guard parameters.count > 0 else { return [] }
    switch parameters[0] {
    case is DynamicGraph.AnyTensor:
      let tensorParameters = parameters as! [DynamicGraph.AnyTensor]
      let graph = tensorParameters[0].graph
      for parameter in tensorParameters {
        assert(parameter.graph === graph)
      }
      // Update private saved_aux.
      let size = Int(ccv_nnc_minimizer_saved_aux_size(minimizer))
      return (0..<(tensorParameters.count * size)).map { _ in graph.variable() }
    case is DynamicGraph.AnyGroup:
      let groupParameters = parameters as! [DynamicGraph.AnyGroup]
      let parallel = groupParameters[0].underlying.count
      precondition(parallel > 0)
      let graph = groupParameters[0].underlying[0].graph
      for group in groupParameters {
        for parameter in group.underlying {
          assert(parameter.graph === graph)
        }
      }
      let size = Int(ccv_nnc_minimizer_saved_aux_size(minimizer))
      return (0..<(groupParameters.count * size)).map { _ in DynamicGraph.Group((0..<parallel).map { _ in graph.variable() }) }
    default:
      fatalError("Cannot support the given type")
    }
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
  public var parameters = [DynamicGraph_AnyParameters]() {
    willSet {
      for var parameter in parameters.compactMap({ $0 as? DynamicGraph_Any }) {
        parameter.requiresGrad = false
      }
    }
    didSet {
      for var parameter in parameters.compactMap({ $0 as? DynamicGraph_Any }) {
        precondition(!parameter.isConstant)
        parameter.requiresGrad = true
      }
      savedAux = savedAux(minimizer: minimizer)
    }
  }

  private var savedAux = [DynamicGraph_Any]()
  private var minimizer: ccv_nnc_cmd_t {
    var params = CmdParamsFactory.factory.newParams()
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
    optimizerStep(graph: graph, minimizer: minimizer, parameters: parameters, savedAux: savedAux, streamContext: streamContext)
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
  public var parameters = [DynamicGraph_AnyParameters]() {
    willSet {
      for var parameter in parameters.compactMap({ $0 as? DynamicGraph_Any }) {
        parameter.requiresGrad = false
      }
    }
    didSet {
      for var parameter in parameters.compactMap({ $0 as? DynamicGraph_Any }) {
        precondition(!parameter.isConstant)
        parameter.requiresGrad = true
      }
      savedAux = savedAux(minimizer: minimizer)
    }
  }

  private var savedAux = [DynamicGraph_Any]()
  private var minimizer: ccv_nnc_cmd_t {
    var params = CmdParamsFactory.factory.newParams()
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
    optimizerStep(graph: graph, minimizer: minimizer, parameters: parameters, savedAux: savedAux, streamContext: streamContext)
  }
}
