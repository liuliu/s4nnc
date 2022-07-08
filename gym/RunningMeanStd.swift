import NNC

public struct RunningMeanStd<TensorElement: DynamicGraph.TensorGroup> {
  public var mean: TensorElement
  public var variance: TensorElement
  public var count: Int
  public init(mean: TensorElement, variance: TensorElement) {
    self.mean = mean
    self.variance = variance
    count = 0
  }
  public mutating func update(_ data: [TensorElement]) {
    let graph = mean.graph
    precondition(data.count >= 1)
    graph.withNoGrad {
      let batchMean: TensorElement
      let batchVar: TensorElement
      if data.count > 1 {
        batchMean = 1 / Float(data.count) * Functional.sum(data)
        batchVar =
          1 / Float(data.count) * Functional.sum(data.map { ($0 - batchMean) .* ($0 - batchMean) })
      } else {
        batchMean = data[0]
        batchVar = graph.variable(like: batchMean)
        batchVar.full(0)
      }
      let delta = batchMean - mean
      let totalCount = count + data.count
      mean = mean + Float(data.count) / Float(totalCount) * delta
      let mA = Float(count) * variance
      let mB = Float(data.count) * batchVar
      let m2 = Functional.sum(
        mA, mB, Float(count) * Float(data.count) / Float(totalCount) * (delta .* delta))
      variance = 1.0 / Float(totalCount) * m2
      count = totalCount
    }
  }
}
