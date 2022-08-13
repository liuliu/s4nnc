func noise<T: RandomNumberGenerator>(_ std: Double, using: inout T) -> Double {
  let u1 = Double.random(in: 0...1, using: &using)
  let u2 = Double.random(in: 0...1, using: &using)
  let mag = std * (-2.0 * .log(u1)).squareRoot()
  return mag * .cos(.pi * 2 * u2)
}
