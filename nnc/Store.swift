import C_fpzip
import C_nnc
import C_zlib
import SQLite3

private let fpzipEncode:
  @convention(c) (
    UnsafeRawPointer?, Int, Int32, UnsafePointer<Int32>?, Int32, UnsafeMutableRawPointer?,
    UnsafeMutableRawPointer?, UnsafeMutablePointer<Int>?, UnsafeMutablePointer<UInt32>?
  ) -> Int32 = {
    data, dataSize, dataType, dimensions, dimensionCount, context, encoded, encodedSize, identifier
    in
    guard dataType == Int32(CCV_64F) || dataType == Int32(CCV_32F) || dataType == Int32(CCV_16F)
    else { return 0 }
    guard let data = data, let dimensions = dimensions, let encoded = encoded,
      let encodedSize = encodedSize, dimensionCount > 0
    else { return 0 }
    guard let fpz = fpzip_write_to_buffer(encoded, encodedSize[0]) else { return 0 }
    defer { fpzip_write_close(fpz) }
    fpz.pointee.type = Int32(FPZIP_TYPE_FLOAT)
    switch dataType {
    case Int32(CCV_64F):
      fpz.pointee.type = Int32(FPZIP_TYPE_DOUBLE)
      fpz.pointee.prec = 64
    case Int32(CCV_32F):
      fpz.pointee.type = Int32(FPZIP_TYPE_FLOAT)
      fpz.pointee.prec = 32
    case Int32(CCV_16F):
      fpz.pointee.type = Int32(FPZIP_TYPE_FLOAT)
      fpz.pointee.prec = 19  // Since we have to retain all exponetial components (i.e. 8, in Float16, that is 5), we need to specify a precision at least 19.
    default:
      return 0
    }
    fpz.pointee.nx = dimensions[Int(dimensionCount) - 1]
    fpz.pointee.ny = dimensionCount >= 2 ? dimensions[Int(dimensionCount) - 2] : 1
    fpz.pointee.nz = dimensionCount >= 3 ? dimensions[Int(dimensionCount) - 3] : 1
    if dimensionCount == 4 {
      fpz.pointee.nf = dimensions[0]
    } else if dimensionCount > 4 {
      var remainingCount = dimensions[Int(dimensionCount) - 4]
      for i in 4..<Int(dimensionCount) {
        remainingCount *= dimensions[Int(dimensionCount) - i - 1]
      }
      fpz.pointee.nf = remainingCount
    }
    fpzip_write_header(fpz)
    var inputData = data
    let f32: UnsafeMutablePointer<Float32>?
    if dataType == Int32(CCV_16F) {
      // Need to do the conversion
      let len =
        Int(fpz.pointee.nx) * Int(fpz.pointee.ny) * Int(fpz.pointee.nz) * Int(fpz.pointee.nf)
      let f32p = UnsafeMutablePointer<Float32>.allocate(capacity: len)
      ccv_half_precision_to_float(
        UnsafeMutableRawPointer(mutating: data).assumingMemoryBound(to: UInt16.self), f32p, len)
      inputData = UnsafeRawPointer(f32p)
      f32 = f32p
    } else {
      f32 = nil
    }
    let outbytes = fpzip_write(fpz, inputData)
    if let f32 = f32 {
      f32.deallocate()
    }
    guard outbytes > 0 && outbytes <= dataSize else { return 0 }
    identifier?[0] = 0xf7217
    encodedSize[0] = outbytes
    return 1
  }

private let fpzipDecode:
  @convention(c) (
    UnsafeRawPointer?, Int, Int32, UnsafePointer<Int32>?, Int32, UInt32, UnsafeMutableRawPointer?,
    UnsafeMutableRawPointer?, UnsafeMutablePointer<Int>?
  ) -> Int32 = {
    data, dataSize, dataType, dimensions, dimensionCount, identifier, context, decoded, decodedSize
    in
    guard identifier == 0xf7217 else { return 0 }
    guard dataType == Int32(CCV_64F) || dataType == Int32(CCV_32F) || dataType == Int32(CCV_16F)
    else { return 0 }
    guard let data = data, let dimensions = dimensions, let decoded = decoded,
      let decodedSize = decodedSize, dimensionCount > 0
    else { return 0 }
    guard let fpz = fpzip_read_from_buffer(data) else { return 0 }
    defer { fpzip_read_close(fpz) }
    guard fpzip_read_header(fpz) != 0 else { return 0 }
    let truncatedCount: Int
    let truncatedLength: Int
    let elementSize: Int
    switch dataType {
    case Int32(CCV_64F):
      guard fpz.pointee.type == Int32(FPZIP_TYPE_DOUBLE) else { return 0 }
      elementSize = MemoryLayout<Double>.size
      truncatedCount = decodedSize[0] / MemoryLayout<Double>.size
      truncatedLength = truncatedCount * MemoryLayout<Double>.size
    case Int32(CCV_32F):
      guard fpz.pointee.type == Int32(FPZIP_TYPE_FLOAT) else { return 0 }
      elementSize = MemoryLayout<Float>.size
      truncatedCount = decodedSize[0] / MemoryLayout<Float>.size
      truncatedLength = truncatedCount * MemoryLayout<Float>.size
    case Int32(CCV_16F):
      guard fpz.pointee.type == Int32(FPZIP_TYPE_FLOAT) else { return 0 }
      elementSize = MemoryLayout<Float>.size
      truncatedCount = decodedSize[0] / MemoryLayout<UInt16>.size
      truncatedLength = truncatedCount * MemoryLayout<UInt16>.size
    default:
      return 0
    }
    let len = Int(fpz.pointee.nx) * Int(fpz.pointee.ny) * Int(fpz.pointee.nz) * Int(fpz.pointee.nf)
    if decodedSize[0] < elementSize * len {
      let buffer = UnsafeMutableRawPointer.allocate(
        byteCount: elementSize * len, alignment: elementSize)
      defer { buffer.deallocate() }
      guard fpzip_read(fpz, buffer) != 0 else { return 0 }
      if dataType == Int32(CCV_16F) {
        ccv_float_to_half_precision(
          buffer.assumingMemoryBound(to: Float.self), buffer.assumingMemoryBound(to: UInt16.self),
          truncatedCount)
      }
      memcpy(decoded, buffer, truncatedLength)
      decodedSize[0] = truncatedLength
      return 1
    } else {
      // Decode directly, and then we do conversion, if it is CCV_16F.
      guard fpzip_read(fpz, decoded) != 0 else { return 0 }
      if dataType == Int32(CCV_16F) {
        ccv_float_to_half_precision(
          decoded.assumingMemoryBound(to: Float.self), decoded.assumingMemoryBound(to: UInt16.self),
          len)
      }
      decodedSize[0] = elementSize * len
    }
    return 1
  }

private let zipEncode:
  @convention(c) (
    UnsafeRawPointer?, Int, Int32, UnsafePointer<Int32>?, Int32, UnsafeMutableRawPointer?,
    UnsafeMutableRawPointer?, UnsafeMutablePointer<Int>?, UnsafeMutablePointer<UInt32>?
  ) -> Int32 = {
    data, dataSize, dataType, dimensions, dimensionCount, context, encoded, encodedSize,
    identifier
    in
    guard let data = data, let dimensions = dimensions, let encoded = encoded,
          let encodedSize = encodedSize, dimensionCount > 0
    else { return 0 }
    guard zip(data: data, dataSize: dataSize, zippedData: encoded, zippedDataSize: encodedSize) else {
      return 0
    }
    identifier?[0] = 0x217
    return 1
  }

private let zipDecode:
  @convention(c) (
    UnsafeRawPointer?, Int, Int32, UnsafePointer<Int32>?, Int32, UInt32, UnsafeMutableRawPointer?,
    UnsafeMutableRawPointer?, UnsafeMutablePointer<Int>?
  ) -> Int32 = {
    data, dataSize, dataType, dimensions, dimensionCount, identifier, context, decoded,
    decodedSize
    in
    guard identifier == 0x217 else { return 0 }
    guard let data = data, let dimensions = dimensions, let decoded = decoded,
      let decodedSize = decodedSize, dimensionCount > 0
    else { return 0 }
    guard unzip(data: data, dataSize: dataSize, unzippedData: decoded, unzippedDataSize: decodedSize) else { return 0 }
    return 1
  }


func truncatedBits(_ number: UInt16, bitCount: UInt16) -> UInt16 {
  guard bitCount > 0 else { return number }
  let mask: UInt16 = (1 << bitCount) - 1
  let discard = number & mask
  let threshold: UInt16 = 1 << (bitCount - 1)
  var shifted = number >> bitCount
  if discard > threshold || (discard == threshold && (shifted & 1) == 1) {
    shifted += 1 // Round to even
  }
  return shifted
}

  // The ezm8 format consists of:
  // |-- zipped exponents size (Int32) --|-- zipped exponents --|-- float without exponent --|
  // Each float without exponent is an 8-bit chunk of data:
  // |-- sign bit --|-- truncated mantissa (7 bits) --|
  // By putting the exponent into its own byte, it seems to make it much easier for zip to compress
  // it well. As for the sign bit and mantissa, they have so far been uncompressible
  private let ezm8Encode:
    @convention(c) (
      UnsafeRawPointer?, Int, Int32, UnsafePointer<Int32>?, Int32, UnsafeMutableRawPointer?,
      UnsafeMutableRawPointer?, UnsafeMutablePointer<Int>?, UnsafeMutablePointer<UInt32>?
    ) -> Int32 = {
      data, dataSize, dataType, dimensions, dimensionCount, context, encoded, encodedSize,
      identifier
      in
      guard let data = data, let dimensions = dimensions, let encoded = encoded,
        let encodedSize = encodedSize, dimensionCount > 0
      else { return 0 }
      guard dataType == Int32(CCV_16F) else { return 0 }
      let floatCount = dataSize / MemoryLayout<Float16>.size
      let floatBytesBuffer = data.assumingMemoryBound(to: UInt16.self)
      let floatsWithoutExp = UnsafeMutablePointer<UInt8>.allocate(capacity: floatCount)
      let exponents = UnsafeMutablePointer<UInt8>.allocate(capacity: floatCount)
      defer {
        exponents.deallocate()
        floatsWithoutExp.deallocate()
      }
      for i in 0..<floatCount {
        let floatBytes = floatBytesBuffer[i]
        let exponent = UInt8((floatBytes >> 10) & ((1 << 5) - 1))
        let signBit = UInt8(floatBytes >> 15)
        let mantissa = floatBytes & ((1 << 10) - 1)
        let truncatedMantissa = UInt8(truncatedBits(mantissa, bitCount: 3))
        exponents[i] = UInt8(exponent)
        floatsWithoutExp[i] = (signBit << 7) | truncatedMantissa
      }
      guard encodedSize[0] > 4 else { return 0 }
      var zippedDataSize = encodedSize[0] - 4
      guard zip(data: exponents,
                dataSize: floatCount,
                zippedData: encoded.advanced(by: 4),
                zippedDataSize: &zippedDataSize) else { return 0 }
      encoded.assumingMemoryBound(to: Int32.self)[0] = Int32(zippedDataSize)
      guard 4 + zippedDataSize + floatCount <= encodedSize[0] else { return 0 }
      memcpy(encoded.advanced(by: 4 + zippedDataSize), floatsWithoutExp, floatCount)
      identifier?[0] = 0x511
      encodedSize[0] =
        4 /* for compressed exponents size */
        + zippedDataSize /* exponents */
        + floatCount /* floats without exponent */
      return 1
    }

private let ezm8Decode:
  @convention(c) (
    UnsafeRawPointer?, Int, Int32, UnsafePointer<Int32>?, Int32, UInt32, UnsafeMutableRawPointer?,
    UnsafeMutableRawPointer?, UnsafeMutablePointer<Int>?
  ) -> Int32 = {
    data, dataSize, dataType, dimensions, dimensionCount, identifier, context, decoded, decodedSize
    in
    guard dataType == Int32(CCV_16F) else { return 0 }
    guard identifier == 0x511 else { return 0 }
    guard let data = data, let dimensions = dimensions, let decoded = decoded,
      let decodedSize = decodedSize, dimensionCount > 0
    else { return 0 }
    let floatCount = decodedSize[0] / MemoryLayout<Float16>.size
    let exponentZipSize = Int(data.assumingMemoryBound(to: Int32.self)[0])
    let exponentZipData = data.advanced(by: MemoryLayout<Int32>.size)
    let exponentBuffer = UnsafeMutablePointer<UInt8>.allocate(capacity: floatCount)
    defer { exponentBuffer.deallocate() }
    var unzippedDataSize = floatCount
    guard unzip(data: exponentZipData,
                dataSize: exponentZipSize,
                unzippedData: exponentBuffer,
                unzippedDataSize: &unzippedDataSize) else { return 0 }
    let floatsWithoutExp = exponentZipData.advanced(by: exponentZipSize).assumingMemoryBound(to: UInt8.self)
    let decodedAsInts = decoded.assumingMemoryBound(to: UInt16.self)
    for i in 0..<floatCount {
      let floatWithoutExp = UInt16(floatsWithoutExp[i])
      let signBit = floatWithoutExp >> 7
      let mantissa = (floatWithoutExp & 0x7f) << 3
      let exponent = UInt16(exponentBuffer[i])
      decodedAsInts[i] = (signBit << 15) | (exponent << 10) | mantissa
    }

    return 1
  }

#if canImport(Compression)
  import Compression

  func zip(data: UnsafeRawPointer, dataSize: Int, zippedData: UnsafeMutableRawPointer, zippedDataSize: UnsafeMutablePointer<Int>) -> Bool {
    let outputSize = compression_encode_buffer(
      zippedData.assumingMemoryBound(to: UInt8.self), zippedDataSize[0],
      data.assumingMemoryBound(to: UInt8.self), dataSize, nil, COMPRESSION_ZLIB)
    guard outputSize > 0 else { return false }
    zippedDataSize[0] = outputSize
    return true
  }

  private func unzip(data: UnsafeRawPointer, dataSize: Int, unzippedData: UnsafeMutableRawPointer, unzippedDataSize: UnsafeMutablePointer<Int>) -> Bool {
    let nextIn = data.assumingMemoryBound(to: UInt8.self)
    let nextOut = unzippedData.assumingMemoryBound(to: UInt8.self)
    var stream = compression_stream(
      dst_ptr: nextOut, dst_size: unzippedDataSize[0], src_ptr: nextIn, src_size: dataSize, state: nil)
    var status = compression_stream_init(&stream, COMPRESSION_STREAM_DECODE, COMPRESSION_ZLIB)
    guard status != COMPRESSION_STATUS_ERROR else { return false }
    defer { compression_stream_destroy(&stream) }
    stream.src_ptr = nextIn
    stream.src_size = dataSize
    stream.dst_ptr = nextOut
    stream.dst_size = unzippedDataSize[0]
    repeat {
      status = compression_stream_process(&stream, Int32(COMPRESSION_STREAM_FINALIZE.rawValue))
      guard status != COMPRESSION_STATUS_ERROR else { return false }
    } while status == COMPRESSION_STATUS_OK && stream.dst_size > 0
    unzippedDataSize[0] = unzippedDataSize[0] - stream.dst_size
    return true
  }

#else

  private func zip(data: UnsafeRawPointer, dataSize: Int, zippedData: UnsafeMutablePointer<Int>, zippedDataSize: UnsafeMutablePointer<Int>) -> Bool {
      var stream = z_stream()
      let streamSize = Int32(MemoryLayout<z_stream>.size)
      let result = deflateInit2_(
        &stream, Z_DEFAULT_COMPRESSION, Z_DEFLATED, -MAX_WBITS, 9, Z_DEFAULT_STRATEGY, ZLIB_VERSION,
        streamSize)
      defer { deflateEnd(&stream) }
      guard result == Z_OK else { return false }
      let chunkSize = 0x8000_0000
      var availableSize = dataSize
      var outputSize = 0
      var availableOutputSize = zippedDataSize[0]
      var flush = Z_NO_FLUSH
      var nextIn = UnsafeMutablePointer<UInt8>(mutating: data.assumingMemoryBound(to: UInt8.self))
      var nextOut = zippedData.assumingMemoryBound(to: UInt8.self)
      repeat {
        let bufferInputSize = availableSize > chunkSize ? chunkSize : availableSize
        stream.next_in = nextIn
        stream.avail_in = UInt32(bufferInputSize)
        flush = availableSize > chunkSize ? Z_NO_FLUSH : Z_FINISH
        repeat {
          stream.next_out = nextOut
          let bufferOutputSize = availableOutputSize > chunkSize ? chunkSize : availableOutputSize
          stream.avail_out = UInt32(bufferOutputSize)
          guard deflate(&stream, flush) >= Z_OK else { return false }
          let thisOutputSize = bufferOutputSize - Int(stream.avail_out)
          nextOut = nextOut.advanced(by: thisOutputSize)
          outputSize += thisOutputSize
          availableOutputSize -= thisOutputSize
        } while stream.avail_out == 0
        nextIn = nextIn.advanced(by: bufferInputSize)
        availableSize -= bufferInputSize
      } while flush != Z_FINISH
      zippedDataSize[0] = outputSize
      return true
    }

  private func unzip(data: UnsafeRawPointer, dataSize: Int, unzippedData: UnsafeMutableRawPointer, unzippedDataSize: UnsafeMutablePointer<Int>) -> Bool {
    var stream = z_stream()
    let streamSize = Int32(MemoryLayout<z_stream>.size)
    var result = inflateInit2_(&stream, -MAX_WBITS, ZLIB_VERSION, streamSize)
    defer { inflateEnd(&stream) }
    guard result == Z_OK else { return false }
    let chunkSize = 0x8000_0000
    var availableSize = dataSize
    var outputSize = 0
    var availableOutputSize = unzippedDataSize[0]
    var nextIn = UnsafeMutablePointer<UInt8>(mutating: data.assumingMemoryBound(to: UInt8.self))
    var nextOut = unzippedData.assumingMemoryBound(to: UInt8.self)
    repeat {
      let bufferInputSize = availableSize > chunkSize ? chunkSize : availableSize
      stream.next_in = nextIn
      stream.avail_in = UInt32(bufferInputSize)
      repeat {
        stream.next_out = nextOut
        let bufferOutputSize = availableOutputSize > chunkSize ? chunkSize : availableOutputSize
        stream.avail_out = UInt32(bufferOutputSize)
        result = inflate(&stream, Z_NO_FLUSH)
        guard result != Z_NEED_DICT && result != Z_DATA_ERROR && result != Z_MEM_ERROR else {
          return false
        }
        let thisOutputSize = bufferOutputSize - Int(stream.avail_out)
        nextOut = nextOut.advanced(by: thisOutputSize)
        outputSize += thisOutputSize
        availableOutputSize -= thisOutputSize
      } while stream.avail_out == 0 && availableOutputSize > 0
      nextIn = nextIn.advanced(by: bufferInputSize)
      availableSize -= bufferInputSize
    } while result != Z_STREAM_END && availableOutputSize > 0
    return true
  }
#endif

private let fpzipAndZipEncode:
  @convention(c) (
    UnsafeRawPointer?, Int, Int32, UnsafePointer<Int32>?, Int32, UnsafeMutableRawPointer?,
    UnsafeMutableRawPointer?, UnsafeMutablePointer<Int>?, UnsafeMutablePointer<UInt32>?
  ) -> Int32 = {
    data, dataSize, dataType, dimensions, dimensionCount, context, encoded, encodedSize, identifier
    in
    // Floating point to use fpzip
    if dataType == Int32(CCV_64F) || dataType == Int32(CCV_32F) || dataType == Int32(CCV_16F) {
      return fpzipEncode(
        data, dataSize, dataType, dimensions, dimensionCount, context, encoded, encodedSize,
        identifier)
    }
    return zipEncode(
      data, dataSize, dataType, dimensions, dimensionCount, context, encoded, encodedSize,
      identifier)
  }

private let fpzipAndZipDecode:
  @convention(c) (
    UnsafeRawPointer?, Int, Int32, UnsafePointer<Int32>?, Int32, UInt32, UnsafeMutableRawPointer?,
    UnsafeMutableRawPointer?, UnsafeMutablePointer<Int>?
  ) -> Int32 = {
    data, dataSize, dataType, dimensions, dimensionCount, identifier, context, decoded, decodedSize
    in
    switch identifier {
    case 0xf7217:
      return fpzipDecode(
        data, dataSize, dataType, dimensions, dimensionCount, identifier, context, decoded,
        decodedSize)
    case 0x217:
      return zipDecode(
        data, dataSize, dataType, dimensions, dimensionCount, identifier, context, decoded,
        decodedSize)
    case 0x511:
      return ezm8Decode(
        data, dataSize, dataType, dimensions, dimensionCount, identifier, context, decoded,
        decodedSize)
    default:
      return 0
    }
  }

extension DynamicGraph {

  final class _Store {
    let sqlite: UnsafeMutableRawPointer
    let flags: Store.OpenFlag
    init(sqlite: OpaquePointer, flags: Store.OpenFlag) {
      self.sqlite = UnsafeMutableRawPointer(sqlite)
      self.flags = flags
    }
    deinit {
      // If the database is opened with WAL mode, this makes sure everything write back to the main
      // database, much easier to operate without worrying the data left in the wal log.
      if flags.contains(.truncateWhenClose) {
        sqlite3_wal_checkpoint_v2(OpaquePointer(sqlite), nil, SQLITE_CHECKPOINT_TRUNCATE, nil, nil)
      }
      sqlite3_close(OpaquePointer(sqlite))
    }
  }

  /**
   * A key-value based parameter store.
   */
  public struct Store {
    public struct OpenFlag: OptionSet {
      public let rawValue: Int
      public init(rawValue: Int) {
        self.rawValue = rawValue
      }
      public static let truncateWhenClose = OpenFlag(rawValue: 1 << 0)
      public static let readOnly = OpenFlag(rawValue: 1 << 1)
    }
    public struct Codec: OptionSet {
      public let rawValue: Int
      public init(rawValue: Int) {
        self.rawValue = rawValue
      }
      public static let fpzip = Codec(rawValue: 1 << 0)
      public static let zip = Codec(rawValue: 1 << 1)
      public static let ezm8 = Codec(rawValue: 1 << 2)
      var encode:
        (
          @convention(c) (
            UnsafeRawPointer?, Int, Int32, UnsafePointer<Int32>?, Int32, UnsafeMutableRawPointer?,
            UnsafeMutableRawPointer?, UnsafeMutablePointer<Int>?, UnsafeMutablePointer<UInt32>?
          ) -> Int32
        )?
      {
        if contains(.ezm8) {
          // .ezm8 is not supported with other formats
          guard self == .ezm8 else { return nil } // TODO: do we want to handle this error differently?
          return ezm8Encode
        } else if contains(.fpzip) && contains(.zip) {
          return fpzipAndZipEncode
        } else if contains(.fpzip) {
          return fpzipEncode
        } else if contains(.zip) {
          return zipEncode
        }
        return nil
      }
      var decode:
        (
          @convention(c) (
            UnsafeRawPointer?, Int, Int32, UnsafePointer<Int32>?, Int32, UInt32,
            UnsafeMutableRawPointer?, UnsafeMutableRawPointer?, UnsafeMutablePointer<Int>?
          ) -> Int32
        )?
      {
        if contains(.ezm8) {
          // .ezm8 is not supported with other formats
          guard self == .ezm8 else { return nil } // TODO: do we want to handle this error differently?
          return ezm8Decode
        } else if contains(.fpzip) && contains(.zip) {
          return fpzipAndZipDecode
        } else if contains(.fpzip) {
          return fpzipDecode
        } else if contains(.zip) {
          return zipDecode
        }
        return nil
      }
    }
    private let graph: DynamicGraph
    private let store: _Store

    /**
     * Read a type-erased tensor from the store.
     *
     * - Parameter key: The key corresponding to that particular tensor.
     */
    public func read(_ key: String, codec: Codec = []) -> NNC.AnyTensor? {
      var underlying: UnsafeMutablePointer<ccv_nnc_tensor_t>? = nil
      let result: Int32
      if codec.isEmpty {
        result = ccv_nnc_tensor_read(store.sqlite, key, nil, nil, &underlying)
      } else {
        var option = ccv_nnc_tensor_io_option_t()
        option.decode = codec.decode
        result = ccv_nnc_tensor_read(store.sqlite, key, nil, &option, &underlying)
      }
      guard result == CCV_IO_FINAL else { return nil }
      let anyTensor = AnyTensorStorage(underlying!)
      return anyTensor.toAnyTensor()
    }

    /**
     * Retrieve codec for a particular key. It must be a tensor to make sense of this.
     *
     * - Parameters:
     *   - key: The key corresponding to a particular tensor.
     * - Returns the codec, if the tensor doesn't exist, return nil.
     */
    public func codec(for key: String) -> Codec? {
      var selectCodec: OpaquePointer? = nil
      sqlite3_prepare_v2(
        OpaquePointer(store.sqlite), "SELECT type FROM tensors WHERE name=?1", -1, &selectCodec, nil
      )
      let SQLITE_TRANSIENT = unsafeBitCast(
        OpaquePointer(bitPattern: -1), to: sqlite3_destructor_type.self)
      sqlite3_bind_text(selectCodec, 1, key, -1, SQLITE_TRANSIENT)
      let codec: Codec?
      if sqlite3_step(selectCodec) == SQLITE_ROW {
        let type = sqlite3_column_int64(selectCodec, 0)
        let identifier = (type >> 32) & 0xffff_ffff
        switch identifier {
        case 0x217:
          codec = .zip
        case 0xf7217:
          codec = .fpzip
        case 0x511:
          codec = .ezm8
        default:
          codec = []
        }
      } else {
        codec = nil
      }
      sqlite3_finalize(selectCodec)
      return codec
    }

    /**
     * Read a tensor from the store into tensor variable from dynamic graph.
     *
     * - Parameters:
     *   - key: The key corresponding to that particular tensor.
     *   - variable: The tensor variable to be initialized with.
     * - Returns whether we successfully initialized the tensor variable.
     */
    @discardableResult
    public func read(_ key: String, variable: DynamicGraph_Any, codec: Codec = []) -> Bool {
      switch variable {
      case let tensor as DynamicGraph.AnyTensor:
        assert(tensor.graph === graph)
        let _graph = graph.cGraph
        let _tensor = tensor._tensor
        let raw = ccv_nnc_tensor_from_variable_impl(_graph, _tensor, nil)
        if raw != nil {
          var underlying = raw
          let result: Int32
          if codec.isEmpty {
            result = ccv_nnc_tensor_read(store.sqlite, key, nil, nil, &underlying)
          } else {
            var option = ccv_nnc_tensor_io_option_t()
            option.decode = codec.decode
            result = ccv_nnc_tensor_read(store.sqlite, key, nil, &option, &underlying)
          }
          if result == CCV_IO_FINAL {
            assert(underlying == raw)
          }
          return result == CCV_IO_FINAL
        }
        var underlying: UnsafeMutablePointer<ccv_nnc_tensor_t>? = nil
        let result: Int32
        if codec.isEmpty {
          result = ccv_nnc_tensor_read(store.sqlite, key, nil, nil, &underlying)
        } else {
          var option = ccv_nnc_tensor_io_option_t()
          option.decode = codec.decode
          result = ccv_nnc_tensor_read(store.sqlite, key, nil, &option, &underlying)
        }
        guard result == CCV_IO_FINAL else { return false }
        let anyTensor = AnyTensorStorage(underlying!)
        ccv_nnc_tensor_variable_set(_graph, _tensor, underlying)
        // Retain the tensor until we freed the variable.
        ccv_nnc_tensor_variable_destructor_hook(
          _graph, _tensor,
          { _, _, ctx in
            // No longer need to retain the tensor.
            Unmanaged<NNC.AnyTensorStorage>.fromOpaque(ctx!).release()
          }, Unmanaged.passRetained(anyTensor).toOpaque())
      case let group as DynamicGraph.AnyGroup:
        for (i, tensor) in group.untyped.enumerated() {
          guard read("\(key)(\(i))", variable: tensor) else {
            return false
          }
        }
      default:
        fatalError("Cannot recognize the variable")
      }
      return true
    }
    public enum ModelReaderResult {
      /// Continue to load parameter with the given name.
      case `continue`(String)
      /// The parameter is loaded, no futher operation need.
      case final(NNC.AnyTensor)
      /// Nothing is loaded.
      case fail
    }
    class ModelReaderHelper {
      let reader: (String, DataType, TensorFormat, TensorShape) -> ModelReaderResult
      let sqlite: UnsafeMutableRawPointer
      init(
        reader: @escaping (String, DataType, TensorFormat, TensorShape) -> ModelReaderResult,
        sqlite: UnsafeMutableRawPointer
      ) {
        self.reader = reader
        self.sqlite = sqlite
      }
    }
    /**
     * Read parameters into a given model.
     *
     * - Parameters:
     *   - key: The key corresponding to a particular model.
     *   - model: The model to be initialized with parameters from a given key.
     *   - reader: You can customize your reader to load parameter with a different name etc.
     */
    public func read(
      _ key: String, model: Model, codec: Codec = [],
      reader: ((String, DataType, TensorFormat, TensorShape) -> ModelReaderResult)? = nil
    ) {
      guard let reader = reader else {
        if codec.isEmpty {
          ccv_cnnp_model_read(store.sqlite, key, nil, model.cModel)
        } else {
          var option = ccv_nnc_tensor_io_option_t()
          option.decode = codec.decode
          ccv_cnnp_model_read(store.sqlite, key, &option, model.cModel)
        }
        return
      }
      let readerHelper = ModelReaderHelper(reader: reader, sqlite: store.sqlite)
      ccv_cnnp_model_set_io(
        model.cModel,
        { (handle, name, dir, options, tensorOut) -> Int32 in
          let readerHelper = Unmanaged<ModelReaderHelper>.fromOpaque(handle!).takeUnretainedValue()
          let cTensorOut = tensorOut!.pointee
          let params = cTensorOut!.pointee.info
          let result = readerHelper.reader(
            name.map { String(cString: $0) } ?? "", DataType.from(cTensorParams: params),
            TensorFormat.from(cTensorParams: params), TensorShape(dims: params.dim))
          switch result {
          case .final(let tensor):
            precondition(tensor.kind == .CPU)
            let dataSize = ccv_nnc_tensor_data_size(tensor.cTensor.pointee.info)
            ccv_nnc_tensor_swap(cTensorOut, name, dir, tensor.cTensor.pointee.data.ptr, dataSize)
            return Int32(CCV_IO_FINAL)
          case .continue(let name):
            return ccv_nnc_tensor_read(readerHelper.sqlite, name, dir, options, tensorOut)
          case .fail:
            return Int32(CCV_IO_ERROR)
          }
        }, nil)
      let unmanaged = Unmanaged.passRetained(readerHelper)
      if codec.isEmpty {
        ccv_cnnp_model_read(unmanaged.toOpaque(), key, nil, model.cModel)
      } else {
        var option = ccv_nnc_tensor_io_option_t()
        option.decode = codec.decode
        ccv_cnnp_model_read(unmanaged.toOpaque(), key, &option, model.cModel)
      }
      ccv_cnnp_model_set_io(model.cModel, nil, nil)
      unmanaged.release()
    }
    /**
     * Read parameters into a given model builder.
     *
     * - Parameters:
     *   - key: The key corresponding to a particular model.
     *   - model: The model builder to be initialized with parameters from a given key.
     *   - reader: You can customize your reader to load parameter with a different name etc.
     */
    public func read(
      _ key: String, model: AnyModelBuilder, codec: Codec = [],
      reader: ((String, DataType, TensorFormat, TensorShape) -> ModelReaderResult)? = nil
    ) {
      model.read(key, from: store, codec: codec, reader: reader)
    }

    /**
     * Write a tensor to the store.
     *
     * - Parameters:
     *   - key: The key corresponding to a particular tensor.
     *   - tensor: The tensor to be persisted.
     */
    public func write(_ key: String, tensor: NNC.AnyTensor, codec: Codec = []) {
      if codec.isEmpty {
        ccv_nnc_tensor_write(tensor.cTensor, store.sqlite, key, nil)
      } else {
        var option = ccv_nnc_tensor_io_option_t()
        option.encode = codec.encode
        ccv_nnc_tensor_write(tensor.cTensor, store.sqlite, key, &option)
      }
    }
    /**
     * Write a tensor variable to the store.
     *
     * - Parameters:
     *   - key: The key corresponding to a particular tensor.
     *   - variable: The tensor variable to be persisted.
     */
    public func write(_ key: String, variable: DynamicGraph_Any, codec: Codec = []) {
      switch variable {
      case let tensor as DynamicGraph.AnyTensor:
        assert(tensor.graph === graph)
        let _graph = graph.cGraph
        let _tensor = tensor._tensor
        let raw = ccv_nnc_tensor_from_variable_impl(_graph, _tensor, nil)!
        if codec.isEmpty {
          ccv_nnc_tensor_write(raw, store.sqlite, key, nil)
        } else {
          var option = ccv_nnc_tensor_io_option_t()
          option.encode = codec.encode
          ccv_nnc_tensor_write(raw, store.sqlite, key, &option)
        }
      case let group as DynamicGraph.AnyGroup:
        for (i, tensor) in group.untyped.enumerated() {
          write("\(key)(\(i))", variable: tensor)
        }
      default:
        fatalError("Cannot recognize the variable")
      }
    }
    /**
     * Write a model to the store.
     *
     * - Parameters:
     *   - key: The key corresponding to a particular model.
     *   - model: The model where its parameters to be persisted.
     */
    public func write(_ key: String, model: Model, codec: Codec = []) {
      if codec.isEmpty {
        ccv_cnnp_model_write(model.cModel, store.sqlite, key, nil)
      } else {
        var option = ccv_nnc_tensor_io_option_t()
        option.encode = codec.encode
        ccv_cnnp_model_write(model.cModel, store.sqlite, key, &option)
      }
    }
    /**
     * Write a model builder to the store.
     *
     * - Parameters:
     *   - key: The key corresponding to a particular model builder.
     *   - model builder: The model where its parameters to be persisted.
     */
    public func write(_ key: String, model: AnyModelBuilder, codec: Codec = []) {
      write(key, model: model.model!)
    }

    init(_ store: _Store, graph: DynamicGraph) {
      self.store = store
      self.graph = graph
    }

    /**
     * Retrieve a list of all tensors in this file. This reads from the disk
     * and could take some time to finish.
     */
    public var keys: [String] {
      var stmt: OpaquePointer? = nil
      sqlite3_prepare_v2(OpaquePointer(store.sqlite), "SELECT name FROM tensors", -1, &stmt, nil)
      guard let stmt = stmt else { return [] }
      var keys = [String]()
      while SQLITE_ROW == sqlite3_step(stmt) {
        guard let cString = sqlite3_column_text(stmt, 0) else { continue }
        keys.append(String(cString: cString))
      }
      sqlite3_finalize(stmt)
      return keys
    }

    /**
     * Remove one tensor by its key.
     */
    public func remove(_ key: String) {
      var deletion: OpaquePointer? = nil
      sqlite3_prepare_v2(
        OpaquePointer(store.sqlite), "DELETE FROM tensors WHERE name=?1", -1, &deletion, nil)
      let SQLITE_TRANSIENT = unsafeBitCast(
        OpaquePointer(bitPattern: -1), to: sqlite3_destructor_type.self)
      sqlite3_bind_text(deletion, 1, key, -1, SQLITE_TRANSIENT)
      sqlite3_step(deletion)
      sqlite3_finalize(deletion)
    }

    /**
     * Remove all tensors from the store. This also vacuums the store to minimize its size.
     */
    public func removeAll() {
      sqlite3_exec(OpaquePointer(store.sqlite), "DELETE FROM tensors", nil, nil, nil)
      sqlite3_exec(OpaquePointer(store.sqlite), "VACUUM", nil, nil, nil)
    }

  }

  /**
   * Open the store from a file.
   *
   * - Parameters:
   *   - filePath: The file path for the store.
   *   - flags: The flags for the opening store. Default to truncateWhenClose.
   *   - procedure: When the store is open, you can access it from this closure.
   * - Returns: Wether this store can be successfully open or not.
   */
  @discardableResult
  public func openStore(
    _ filePath: String, flags: Store.OpenFlag = .truncateWhenClose,
    procedure: (_ store: Store) throws -> Void
  ) rethrows -> Bool {
    var _sqlite: OpaquePointer? = nil
    if flags.contains(.readOnly) {
      if sqlite3_libversion_number() >= 3_022_000 {
        sqlite3_open_v2(filePath, &_sqlite, SQLITE_OPEN_READONLY, nil)
      } else {  // At least if it is readOnly, we won't create the file.
        sqlite3_open_v2(filePath, &_sqlite, SQLITE_OPEN_READWRITE, nil)
      }
    } else {
      sqlite3_open_v2(filePath, &_sqlite, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nil)
    }
    guard let sqlite = _sqlite else { return false }
    sqlite3_busy_timeout(sqlite, 30_000)  // This is essential to have real-world usages.
    let store = Store(_Store(sqlite: sqlite, flags: flags), graph: self)
    try procedure(store)
    return true
  }

}
