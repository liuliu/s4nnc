import C_ccv
import C_nnc
import Foundation
import NNC

extension SummaryWriter {
  /// Add tensor for tensorboard histograms dashboard.
  public func addHistogram(
    _ tag: String, _ value: Tensor<Float>, step: Int,
    wallTime: Double = Date().timeIntervalSince1970, displayName: String? = nil,
    description: String? = nil
  ) {
    var summaryMetadata = Tensorboard_SummaryMetadata()
    summaryMetadata.displayName = displayName ?? tag
    summaryMetadata.summaryDescription = description ?? ""

    var histogram = Tensorboard_HistogramProto()
    // Only support "tensorflow" style at the moment.
    var limit = [Double]()
    var v: Double = 1e-12
    while v < 1e20 {
      limit.append(v)
      v *= 1.1
    }
    var bucketLimit = limit.reversed().map { -$0 }
    bucketLimit.append(contentsOf: limit)
    bucketLimit.append(.greatestFiniteMagnitude)
    bucketLimit.append(.greatestFiniteMagnitude)
    assert(bucketLimit.count == 775 * 2)
    histogram.bucketLimit = bucketLimit
    let histoTensor = Tensor<Int32>(.CPU, .C(bucketLimit.count))
    let statsTensor = Tensor<Float>(.CPU, .C(4))
    let vTensor = value.kind == .CPU ? value : value.toCPU()  // Move to CPU if needed.
    withExtendedLifetime(vTensor) {
      var input: UnsafeMutablePointer<ccv_nnc_tensor_t>? = vTensor.cTensor
      let outputs = UnsafeMutablePointer<UnsafeMutablePointer<ccv_nnc_tensor_t>?>.allocate(
        capacity: 2)
      outputs[0] = histoTensor.cTensor
      outputs[1] = statsTensor.cTensor
      var params = CmdParamsFactory.factory.newParams()
      params.histogram.type = Int32(CCV_NNC_HISTOGRAM_LOGARITHMIC)
      params.histogram.min = 1e-12
      params.histogram.max = 1e20
      params.histogram.rate = 1.1
      ccv_nnc_cmd_exec(
        ccv_nnc_cmd(
          CCV_NNC_HISTOGRAM_FORWARD, nil, params, 0),
        ccv_nnc_no_hint, 0, &input, 1, outputs, 2, nil)
      outputs.deallocate()
    }
    histogram.min = Double(statsTensor[0])
    histogram.max = Double(statsTensor[1])
    histogram.sum = Double(statsTensor[2])
    histogram.sumSquares = Double(statsTensor[3])
    histogram.num = Double(value.dimensions.reduce(1, *))
    var bucket: [Double] = Array(repeating: 0, count: bucketLimit.count)
    for i in 0..<bucket.count {
      bucket[i] = Double(histoTensor[i])
    }
    histogram.bucket = bucket

    var summaryValue = Tensorboard_Summary.Value()
    summaryValue.tag = tag
    summaryValue.histo = histogram
    summaryValue.metadata = summaryMetadata

    var summary = Tensorboard_Summary()
    summary.value = [summaryValue]

    var event = Tensorboard_Event()
    event.summary = summary
    event.wallTime = wallTime
    event.step = Int64(step)
    do {
      try eventLogger.add(event)
    } catch {
      fatalError("Could not add \(event) to log: \(error)")
    }
  }

  /// Add tensor for tensorboard histograms dashboard.
  public func addHistogram(
    _ tag: String, _ value: DynamicGraph.Tensor<Float>, step: Int,
    wallTime: Double = Date().timeIntervalSince1970, displayName: String? = nil,
    description: String? = nil
  ) {
    addHistogram(
      tag, value.rawValue, step: step, wallTime: wallTime, displayName: displayName,
      description: description)
  }
}

extension SummaryWriter {
  /// Add tensor for tensorboard images dashboard.
  public func addImage(
    _ tag: String, _ value: Tensor<Float>, step: Int,
    wallTime: Double = Date().timeIntervalSince1970, displayName: String? = nil,
    description: String? = nil
  ) {
    var summaryMetadata = Tensorboard_SummaryMetadata()
    summaryMetadata.displayName = displayName ?? tag
    summaryMetadata.summaryDescription = description ?? ""

    var image = Tensorboard_Summary.Image()
    let vTensor = value.kind == .CPU ? value : value.toCPU()  // Move to CPU if needed.
    let dimensions = vTensor.dimensions
    let width: Int
    let height: Int
    let channel: Int
    switch dimensions.count {
    case 1:
      width = dimensions[0]
      height = 1
      channel = 1
    case 2:
      height = dimensions[0]
      width = dimensions[1]
      channel = 1
    case 3...:
      switch vTensor.format {
      case .NHWC:
        height = dimensions[dimensions.count - 3]
        width = dimensions[dimensions.count - 2]
        channel = dimensions[dimensions.count - 1]
        break
      case .NCHW:
        channel = dimensions[dimensions.count - 3]
        height = dimensions[dimensions.count - 2]
        width = dimensions[dimensions.count - 1]
        break
      default:
        fatalError("Unsupported tensor \(vTensor)")
      }
    default:
      fatalError("Unsupported dimension of tensor \(vTensor)")
    }
    precondition(channel <= 4)
    image.width = Int32(width)
    image.height = Int32(height)
    var fTensor: Tensor<Float> = Tensor(.CPU, format: .NHWC, dimensions: [height, width, channel])
    if vTensor.format == .NCHW {  // Need to convert to .NHWC format.
      fTensor[...] = vTensor.reshaped(.CHW(channel, height, width))
    } else {
      fTensor[...] = vTensor.reshaped(.HWC(height, width, channel))
    }
    var output: UnsafeMutableRawPointer? = nil
    withExtendedLifetime(fTensor) {
      let input: UnsafeMutablePointer<ccv_dense_matrix_t> = UnsafeMutableRawPointer(fTensor.cTensor)
        .assumingMemoryBound(to: ccv_dense_matrix_t.self)
      ccv_scale(input, &output, Int32(CCV_8U), 255)  // Scale to 255 range.
    }
    image.colorspace = Int32(channel)
    if let output = output {
      let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: width * height * channel)
      var count = width * height * channel
      ccv_write(
        output.assumingMemoryBound(to: ccv_dense_matrix_t.self), buffer, &count,
        Int32(CCV_IO_PNG_STREAM), nil)
      ccv_matrix_free(output)
      image.encodedImageString = Data(bytesNoCopy: buffer, count: count, deallocator: .free)
    }

    var summaryValue = Tensorboard_Summary.Value()
    summaryValue.tag = tag
    summaryValue.image = image
    summaryValue.metadata = summaryMetadata

    var summary = Tensorboard_Summary()
    summary.value = [summaryValue]

    var event = Tensorboard_Event()
    event.summary = summary
    event.wallTime = wallTime
    event.step = Int64(step)
    do {
      try eventLogger.add(event)
    } catch {
      fatalError("Could not add \(event) to log: \(error)")
    }
  }

  /// Add tensor for tensorboard images dashboard.
  public func addImage(
    _ tag: String, _ value: DynamicGraph.Tensor<Float>, step: Int,
    wallTime: Double = Date().timeIntervalSince1970, displayName: String? = nil,
    description: String? = nil
  ) {
    addImage(
      tag, value.rawValue, step: step, wallTime: wallTime, displayName: displayName,
      description: description)
  }
}

extension SummaryWriter {
  /// Add graph for tensorboard graphs dashboard.
  public func addGraph(
    _ tag: String, _ value: Model, step: Int,
    wallTime: Double = Date().timeIntervalSince1970, displayName: String? = nil,
    description: String? = nil
  ) {
  }

  /// Add graph for tensorboard graphs dashboard.
  public func addGraph(
    _ tag: String, _ value: DynamicGraph, step: Int,
    wallTime: Double = Date().timeIntervalSince1970, displayName: String? = nil,
    description: String? = nil
  ) {
  }
}
