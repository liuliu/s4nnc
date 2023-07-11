import C_nnc

public struct GradScaler {
  public var scale: Float
  public var growthFactor: Float
  public var backoffFactor: Float
  public var growthInterval: Int
  private var unscaled: Bool
  private var step: Int
  public init(
    scale: Float = 65536, growthFactor: Float = 2, backoffFactor: Float = 0.5,
    growthInterval: Int = 2_000
  ) {
    self.scale = scale
    self.growthFactor = growthFactor
    self.backoffFactor = backoffFactor
    self.growthInterval = growthInterval
    unscaled = false
    step = 0
  }
  public func scale<T: DynamicGraph.TensorGroup>(_ loss: T, streamContext: StreamContext? = nil)
    -> T
  {
    var loss = loss
    let graph = loss.graph
    guard let grad = loss.grad else {
      let grad = graph.constant(like: loss)
      grad.full(scale)
      loss.grad = grad.typeErased
      return loss
    }
    loss.grad = T(grad).scaled(by: scale, streamContext: streamContext).typeErased
    return loss
  }
  private func isNaN<T: Optimizer>(_ optimizers: [T], streamContext: StreamContext? = nil) -> Bool {
    precondition(optimizers.count > 0)
    let graph = optimizers[0].graph
    let _streamContext = (streamContext ?? graph.streamContext)?._stream
    let parameters = optimizers.flatMap { $0.parameters }
    let modelParameters = parameters.compactMap { $0 as? Model.Parameters }
    let primaryModelParameters = modelParameters.filter {
      $0._io == $0.model!._parameters && $0.model!.owner == nil
    }
    var models = Set(modelParameters.map { HashableModel(model: $0.model!.owner ?? $0.model!) })
    // Reset these models with primary parameters with the new minimizer.
    for parameter in primaryModelParameters {
      let model = parameter.model!
      models.remove(HashableModel(model: model))
      if ccv_cnnp_model_parameter_gradients_isnan(model.cModel, parameter._io, _streamContext) != 0
      {
        return true
      }
    }
    // Set minimizers on other models.
    var modelParametersMap = [HashableModel: [Model.Parameters]]()
    for parameter in modelParameters
    // If parameter is not primary
    where parameter._io != parameter.model!._parameters || parameter.model!.owner != nil {
      let model = parameter.model!.owner ?? parameter.model!
      let index = HashableModel(model: model)
      guard models.contains(index) else { continue }
      modelParametersMap[index, default: [Model.Parameters]()].append(parameter)
    }
    for (key, parameters) in modelParametersMap {
      for parameter in parameters {
        if ccv_cnnp_model_parameter_gradients_isnan(key.model.cModel, parameter._io, _streamContext)
          != 0
        {
          return true
        }
      }
    }
    let tensorParameters = parameters.compactMap { $0 as? DynamicGraph.AnyTensor }
    for tensor in tensorParameters {
      if tensor.grad?.isNaN ?? false {
        return true
      }
    }
    let groupParameters = parameters.compactMap { $0 as? DynamicGraph.Group }
    for tensor in groupParameters {
      if tensor.grad?.isNaN ?? false {
        return true
      }
    }
    return false
  }
  public mutating func unscale<T: Optimizer>(_ optimizers: [T], streamContext: StreamContext? = nil)
  {
    precondition(optimizers.count > 0)
    let graph = optimizers[0].graph
    let _streamContext = (streamContext ?? graph.streamContext)?._stream
    let parameters = optimizers.flatMap { $0.parameters }
    let modelParameters = parameters.compactMap { $0 as? Model.Parameters }
    let primaryModelParameters = modelParameters.filter {
      $0._io == $0.model!._parameters && $0.model!.owner == nil
    }
    var models = Set(modelParameters.map { HashableModel(model: $0.model!.owner ?? $0.model!) })
    var params = CmdParamsFactory.factory.newParams()
    params.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    params.blas.a = (1.0 / scale, 0, 0)
    let scalmul = ccv_nnc_cmd(CCV_NNC_SCALAR_MUL_FORWARD, nil, params, 0)
    // Reset these models with primary parameters with the new minimizer.
    for parameter in primaryModelParameters {
      let model = parameter.model!
      models.remove(HashableModel(model: model))
      ccv_cnnp_model_parameter_gradients_map(
        model.cModel, parameter._io, scalmul, ccv_nnc_no_hint, 0, nil, 0, nil, 0, _streamContext)
    }
    // Set minimizers on other models.
    var modelParametersMap = [HashableModel: [Model.Parameters]]()
    for parameter in modelParameters
    // If parameter is not primary
    where parameter._io != parameter.model!._parameters || parameter.model!.owner != nil {
      let model = parameter.model!.owner ?? parameter.model!
      let index = HashableModel(model: model)
      guard models.contains(index) else { continue }
      modelParametersMap[index, default: [Model.Parameters]()].append(parameter)
    }
    for (key, parameters) in modelParametersMap {
      for parameter in parameters {
        ccv_cnnp_model_parameter_gradients_map(
          key.model.cModel, parameter._io, scalmul, ccv_nnc_no_hint, 0, nil, 0, nil, 0,
          _streamContext)
      }
    }
    let tensorParameters = parameters.compactMap { $0 as? DynamicGraph.AnyTensor }
    for tensor in tensorParameters {
      Functional.exec(
        cmd: scalmul, hint: ccv_nnc_no_hint, inputs: tensor, outputs: [tensor],
        streamContext: streamContext)
    }
    let groupParameters = parameters.compactMap { $0 as? DynamicGraph.Group }
    for tensor in groupParameters {
      Functional.exec(
        cmd: scalmul, hint: ccv_nnc_no_hint, inputs: tensor, outputs: [tensor],
        streamContext: streamContext)
    }
    unscaled = true
  }
  public mutating func step<T: Optimizer>(
    _ optimizers: inout [T], streamContext: StreamContext? = nil
  ) {
    guard !isNaN(optimizers, streamContext: streamContext) else {
      // Running optimizer step with noop. This way, we will clean up the apply gradients thing.
      let params = CmdParamsFactory.factory.newParams()
      let noop = ccv_nnc_cmd(CCV_NNC_NOOP, nil, params, 0)
      for optimizer in optimizers {
        let addons = (optimizer as! OptimizerAddons)
        optimizerStep(
          graph: optimizer.graph, minimizer: noop, parameters: optimizer.parameters,
          savedAux: addons.savedAux, streamContext: streamContext)
      }
      scale *= backoffFactor
      return
    }
    step += 1
    if step >= growthInterval {
      scale *= growthFactor
      step = 0
    }
    guard !unscaled else {
      optimizers.step(streamContext: streamContext)
      return
    }
    precondition(optimizers.count > 0)
    let unscale = 1.0 / scale
    var unscaledOptimizers: [T] = optimizers.map {
      var optimizer = $0
      optimizer.scale *= unscale
      return optimizer
    }
    unscaledOptimizers.step(streamContext: streamContext)
    for (i, optimizer) in optimizers.enumerated() {
      if var optimizer = optimizer as? OptimizerTrackSteps {
        optimizer.step += 1
        optimizers[i] = optimizer as! T
      }
    }
    unscaled = false
  }
  public mutating func unscale<T: Optimizer>(_ optimizer: T, streamContext: StreamContext? = nil) {
    unscale([optimizer], streamContext: streamContext)
  }
  public mutating func step<T: Optimizer>(_ optimizer: inout T, streamContext: StreamContext? = nil)
  {
    var optimizers = [optimizer]
    step(&optimizers, streamContext: streamContext)
    optimizer = optimizers[0]
  }
}
