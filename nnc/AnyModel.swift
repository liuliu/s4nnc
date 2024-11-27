
public protocol AnyModel {
  /**
   * Whether the existing model is for testing or training.
   */
  var testing: Bool { get set }
  /**
   * Whether to enable memory reduction for this model. The current supported memory reduction
   * technique is to redo datatype conversion during backward pass if needed.
   */
  var memoryReduction: Bool { get set }
  /**
   * Specify the maximum number of streams we need to allocate to run this model.
   */
  var maxConcurrency: StreamContext.Concurrency { get set }
  /**
   * Abstract representation of the stateful components from the model builder.
   */
  var parameters: Model.Parameters { get }
  /**
   * Shortcut for weight parameter.
   */
  var weight: Model.Parameters { get }
  /**
   * Shortcut for bias parameter.
   */
  var bias: Model.Parameters { get }
  /**
   * Broadly speaking, you can have two types of parameters, weight and bias.
   * You can get them in abstract fashion with this method.
   *
   * - Parameter type: Whether it is weight or bias.
   * - Returns: An abstract representation of parameters.
   */
  func parameters(for type: Model.ParametersType) -> Model.Parameters
}
