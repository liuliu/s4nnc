public protocol Renderable {
  func render(width: Int, height: Int)
}

extension Renderable {
  // Default to render at 60fps.
  public func render() {
    render(width: 1280, height: 720)
  }
}
