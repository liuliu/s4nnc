import C_nnc

/// A stream context is an object that an execution can be performed upon.
public final class StreamContext {
  public enum Concurrency {
    case noLimit
    case limit(Int)
    init(rawValue: Int) {
      switch rawValue {
      case 0:
        self = .noLimit
      default:
        self = .limit(rawValue)
      }
    }
    var rawValue: Int {
      switch self {
      case .noLimit:
        return 0
      case .limit(let value):
        return value
      }
    }
  }

  let selfOwned: Bool
  let _stream: OpaquePointer

  init(stream: OpaquePointer, selfOwned: Bool) {
    _stream = stream
    self.selfOwned = selfOwned
  }

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
    selfOwned = true
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

  /**
   * Set seed for this particular stream context. If not set, it inherits from the global context.
   */
  public func setSeed(_ seed: UInt32) {
    ccv_nnc_stream_context_set_seed(_stream, seed)
  }

  deinit {
    guard selfOwned else { return }
    ccv_nnc_stream_context_free(_stream)
  }
}
