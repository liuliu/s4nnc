import C_nnc

public final class StreamContext {

  let _stream: OpaquePointer

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

  public func joined() {
    ccv_nnc_stream_context_wait(_stream)
  }

  public func async(_ closure: @escaping () -> Void) {
    ccv_nnc_stream_context_add_callback(_stream, { _, context in
      let closure = Unmanaged<AnyObject>.fromOpaque(context!).takeRetainedValue() as! (() -> Void)
      closure()
    }, Unmanaged.passRetained(closure as AnyObject).toOpaque())
  }

  deinit {
    ccv_nnc_stream_context_free(_stream)
  }
}

