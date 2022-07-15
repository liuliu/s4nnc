public struct NumericalStatistics {
  public var mean: Float
  public var std: Float
  public init(mean: Float = 0, std: Float = 0) {
    self.mean = mean
    self.std = std
  }
  public init<C: Collection>(_ array: C) where C.Element == Float {
    if array.count > 0 {
      let mean = (array.reduce(0) { $0 + $1 }) / Float(array.count)
      self.mean = mean
      self.std = ((array.reduce(0) { $0 + ($1 - mean) * ($1 - mean) }) / Float(array.count))
        .squareRoot()
    } else {
      mean = 0
      std = 0
    }
  }
}
