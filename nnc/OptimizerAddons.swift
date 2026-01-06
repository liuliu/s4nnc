#if canImport(C_nnc)
import C_nnc
#elseif canImport(C_swiftpm_nnc)
import C_swiftpm_nnc
#endif

/// Stochastic gradient descent optimizer.
public struct SGDOptimizer: Optimizer, OptimizerAddons {
  public let graph: DynamicGraph
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

  var savedAux = [DynamicGraph_Any]()
  var minimizer: ccv_nnc_cmd_t {
    var params = CmdParamsFactory.factory.newParams()
    params.sgd.nesterov = nesterov ? 1 : 0
    params.sgd.rate = rate
    params.sgd.scale = scale
    params.sgd.decay = decay
    params.sgd.momentum = momentum
    params.sgd.dampening = dampening
    return ccv_nnc_cmd(CCV_NNC_SGD_FORWARD, nil, params, 0)
  }

  public init(
    _ graph: DynamicGraph, nesterov: Bool, rate: Float, scale: Float, decay: Float, momentum: Float,
    dampening: Float
  ) {
    self.graph = graph
    self.nesterov = nesterov
    self.rate = rate
    self.scale = scale
    self.decay = decay
    self.momentum = momentum
    self.dampening = dampening
  }

  public func step(streamContext: StreamContext?) {
    optimizerStep(
      graph: graph, minimizer: minimizer, parameters: parameters, savedAux: savedAux,
      streamContext: streamContext)
  }
}

/// Adam optimizer.
public struct AdamOptimizer: Optimizer, OptimizerAddons, OptimizerTrackSteps {
  public let graph: DynamicGraph
  public var step: Int
  public var rate: Float
  public var betas: (Float, Float)
  public var decay: Float
  public var epsilon: Float
  public var scale: Float
  public var amsgrad: Bool
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

  var savedAux = [DynamicGraph_Any]()
  var minimizer: ccv_nnc_cmd_t {
    var params = CmdParamsFactory.factory.newParams()
    params.adam.step = Int32(step)
    params.adam.rate = rate
    params.adam.scale = scale
    params.adam.beta1 = betas.0
    params.adam.beta2 = betas.1
    params.adam.decay = decay
    params.adam.epsilon = epsilon
    params.adam.amsgrad = amsgrad ? 1 : 0
    return ccv_nnc_cmd(CCV_NNC_ADAM_FORWARD, nil, params, 0)
  }

  public init(
    _ graph: DynamicGraph, rate: Float = 0.001, step: Int = 1, betas: (Float, Float) = (0.9, 0.999),
    decay: Float = 0, epsilon: Float = 1e-8, amsgrad: Bool = false
  ) {
    self.graph = graph
    self.step = step
    self.rate = rate
    self.betas = betas
    self.decay = decay
    self.epsilon = epsilon
    self.amsgrad = amsgrad
    scale = 1
  }

  public mutating func step(streamContext: StreamContext?) {
    optimizerStep(
      graph: graph, minimizer: minimizer, parameters: parameters, savedAux: savedAux,
      streamContext: streamContext)
    step += 1
  }
}

/// LAMB optimizer.
public struct LAMBOptimizer: Optimizer, OptimizerAddons, OptimizerTrackSteps {
  public let graph: DynamicGraph
  public var step: Int
  public var rate: Float
  public var betas: (Float, Float)
  public var decay: Float
  public var epsilon: Float
  public var scale: Float
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

  var savedAux = [DynamicGraph_Any]()
  var minimizer: ccv_nnc_cmd_t {
    var params = CmdParamsFactory.factory.newParams()
    params.lamb.step = Int32(step)
    params.lamb.rate = rate
    params.lamb.scale = scale
    params.lamb.beta1 = betas.0
    params.lamb.beta2 = betas.1
    params.lamb.decay = decay
    params.lamb.epsilon = epsilon
    return ccv_nnc_cmd(CCV_NNC_LAMB_FORWARD, nil, params, 0)
  }

  public init(
    _ graph: DynamicGraph, rate: Float = 0.001, step: Int = 1, betas: (Float, Float) = (0.9, 0.999),
    decay: Float = 0, epsilon: Float = 1e-6
  ) {
    self.graph = graph
    self.step = step
    self.rate = rate
    self.betas = betas
    self.decay = decay
    self.epsilon = epsilon
    scale = 1
  }

  public mutating func step(streamContext: StreamContext?) {
    optimizerStep(
      graph: graph, minimizer: minimizer, parameters: parameters, savedAux: savedAux,
      streamContext: streamContext)
    step += 1
  }
}

/// AdamW optimizer.
public struct AdamWOptimizer: Optimizer, OptimizerAddons, OptimizerTrackSteps {
  public let graph: DynamicGraph
  public var step: Int
  public var rate: Float
  public var betas: (Float, Float)
  public var decay: Float
  public var epsilon: Float
  public var scale: Float
  public var amsgrad: Bool
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

  var savedAux = [DynamicGraph_Any]()
  var minimizer: ccv_nnc_cmd_t {
    var params = CmdParamsFactory.factory.newParams()
    params.adam.step = Int32(step)
    params.adam.rate = rate
    params.adam.scale = scale
    params.adam.beta1 = betas.0
    params.adam.beta2 = betas.1
    params.adam.decay = decay
    params.adam.epsilon = epsilon
    params.adam.amsgrad = amsgrad ? 1 : 0
    return ccv_nnc_cmd(CCV_NNC_ADAMW_FORWARD, nil, params, 0)
  }

  public init(
    _ graph: DynamicGraph, rate: Float = 0.001, step: Int = 1, betas: (Float, Float) = (0.9, 0.999),
    decay: Float = 0, epsilon: Float = 1e-8, amsgrad: Bool = false
  ) {
    self.graph = graph
    self.step = step
    self.rate = rate
    self.betas = betas
    self.decay = decay
    self.epsilon = epsilon
    self.amsgrad = amsgrad
    scale = 1
  }

  public mutating func step(streamContext: StreamContext?) {
    optimizerStep(
      graph: graph, minimizer: minimizer, parameters: parameters, savedAux: savedAux,
      streamContext: streamContext)
    step += 1
  }
}
