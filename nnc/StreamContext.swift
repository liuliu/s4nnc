import C_nnc

/// A stream context is an object that an execution can be performed upon.
public final class StreamContext {

  let _stream: OpaquePointer

  /**
   * Create a new stream context.
   *
   * - Parameter kind: Whether this stream context is on CPU or GPU.
   */
  public init(_ kind: DeviceKind) {
    let type: Int32
    switch kind {
    case .CPU:
      type = Int32(CCV_STREAM_CONTEXT_CPU)
    case .GPU(let ordinal):
      type = Int32((ordinal << 8) | CCV_STREAM_CONTEXT_GPU)
    }
    _stream = ccv_nnc_stream_context_new(type)!
  }

  /**
   * Wait until all executions on this stream context to finish.
   */
  public func joined() {
    ccv_nnc_stream_context_wait(_stream)
  }

  /**
   * Dispatch a block to be executed when all previous executions prior to
   * this method call are done.
   */
  public func async(_ closure: @escaping () -> Void) {
    ccv_nnc_stream_context_add_callback(
      _stream,
      { context in
        let closure = Unmanaged<AnyObject>.fromOpaque(context!).takeRetainedValue() as! (() -> Void)
        closure()
      }, Unmanaged.passRetained(closure as AnyObject).toOpaque())
  }

  deinit {
    ccv_nnc_stream_context_free(_stream)
  }
}
