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

  deinit {
    ccv_nnc_stream_context_free(_stream)
  }
}

