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

func formatGraph(
  _ graph: OpaquePointer?, _ node: Int32, _ name: UnsafePointer<Int8>?, _ cmd: ccv_nnc_cmd_t,
  _ flags: Int32, _ incomings: UnsafePointer<Int32>?, _ incomingSize: Int32,
  _ outgoings: UnsafePointer<Int32>?, _ outgoingSize: Int32, _ inputs: UnsafePointer<Int32>?,
  _ inputSize: Int32, _ outputs: UnsafePointer<Int32>?, _ outputSize: Int32,
  _ context: UnsafeMutableRawPointer?
) {
  guard let context = context else {
    return
  }
  let graphDef = Unmanaged<SummaryWriter.Graph>.fromOpaque(context).takeUnretainedValue()
  let name = name.map { String(cString: $0) } ?? ""
  var node = SummaryWriter.Graph.Node(
    id: node, name: name, op: cmd.cmd, inputs: [],
    outputs: outputs.map { Array(UnsafeBufferPointer(start: $0, count: Int(outputSize))) } ?? [])
  // Create a map from nodes input to incomings. The mapping is more involved because it needs to
  // take into account alias.
  if let inputs = inputs, let incomings = incomings {
    for i in 0..<Int(inputSize) {
      guard inputs[i] >= 0 else { continue }
      var tensor = ccv_nnc_tensor_symbol_t(d: inputs[i], graph: graph)
      let cTensorParams = ccv_nnc_tensor_symbol_params(graph, tensor)  // There is no exposed method to get name at the moment.
      if ccv_nnc_is_tensor_auto(cTensorParams) == 0 {  // Now I can parse its shape.
        graphDef.tensors[tensor.d] = SummaryWriter.Graph.Tensor(
          id: tensor.d, dimensions: fromCDimensions(cTensorParams.dim),
          dataType: .from(cTensorParams: cTensorParams), kind: .from(cTensorParams: cTensorParams))
      }
      let aliasTo = ccv_nnc_tensor_symbol_alias_to(graph, tensor)
      if aliasTo.d >= 0 {
        tensor = aliasTo
      }
      var input = SummaryWriter.Graph.Input.variable(inputs[i])
      for j in 0..<Int(incomingSize) {
        let exec = ccv_nnc_graph_exec_symbol_t(d: incomings[j], graph: graph)
        var outputs: UnsafePointer<Int32>? = nil
        var outputSize: Int32 = 0
        ccv_nnc_graph_exec_symbol_io(graph, exec, nil, nil, &outputs, &outputSize)
        if let outputs = outputs {
          for k in 0..<Int(outputSize) {
            var output = ccv_nnc_tensor_symbol_t(d: outputs[k], graph: graph)
            let aliasTo = ccv_nnc_tensor_symbol_alias_to(graph, output)
            if aliasTo.d >= 0 {
              output = aliasTo
            }
            if output.d == tensor.d {
              input = .node(exec.d, Int32(k))
              break
            }
          }
          if case .node(_, _) = input {
            break
          }
        }
      }
      node.inputs.append(input)
    }
  }
  graphDef.nodes[node.id] = node
}

extension SummaryWriter {
  final class Graph {
    enum Input {
      case variable(Int32)
      case node(Int32, Int32)  // node id, index to the input.
    }

    struct Node {
      var id: Int32
      var name: String
      var op: UInt32
      var inputs: [Input]
      var outputs: [Int32]
      var opName: String {
        guard let cName = ccv_nnc_cmd_name(op) else { return "<unknown>" }
        let name = String(cString: cName)
        if name.hasPrefix("CCV_NNC_") {
          return String(name.suffix(from: name.index(name.startIndex, offsetBy: 8)))
        }
        return name
      }
    }

    struct Tensor {
      var id: Int32
      var dimensions: [Int]
      var dataType: DataType
      var kind: DeviceKind
    }

    var tensors = [Int32: Tensor]()
    var nodes = [Int32: Node]()

    func uniqueName() {  // Mutate the nodes array with unique names.
      var nameMap = Set<String>()
      for (id, var node) in nodes {
        if node.name.isEmpty {
          node.name = "node_\(id)"
          nodes[id] = node
        } else if nameMap.contains(node.name) {  // Check if there are duplicate names already.
          var name = "\(node.name)_\(id)"
          var i = 0
          while nameMap.contains(name) {
            name = "\(node.name)_\(id)_\(i)"
            i += 1
          }
          nameMap.insert(name)
          node.name = name
          nodes[id] = node
        }
      }
    }

    var proto: Tensorboard_GraphDef {
      var graphDef = Tensorboard_GraphDef()
      var nodeDefs = [Tensorboard_NodeDef]()
      var variables = Set<Int32>()
      for node in nodes.values {
        for input in node.inputs {
          if case .variable(let id) = input {
            variables.insert(id)
          }
        }
      }
      for variable in variables {
        var nodeDef = Tensorboard_NodeDef()
        nodeDef.name = "tensor_\(variable)"
        nodeDef.op = "Variable"
        nodeDef.input = []
        if let tensor = tensors[variable] {
          var dtype = Tensorboard_AttrValue()
          switch tensor.dataType {
          case .Float32:
            dtype.type = .dtFloat
          case .Float64:
            dtype.type = .dtDouble
          case .Float16:
            dtype.type = .dtHalf
          case .Int64:
            dtype.type = .dtInt64
          case .Int32:
            dtype.type = .dtInt32
          case .UInt8:
            dtype.type = .dtUint8
          }
          var shape = Tensorboard_AttrValue()
          var shapeProto = Tensorboard_TensorShapeProto()
          shapeProto.dim = tensor.dimensions.map {
            var dim = Tensorboard_TensorShapeProto.Dim()
            dim.size = Int64($0)
            return dim
          }
          shape.shape = shapeProto
          nodeDef.attr = ["dtype": dtype, "shape": shape]
        }
        nodeDefs.append(nodeDef)
      }
      for node in nodes.values {
        var nodeDef = Tensorboard_NodeDef()
        nodeDef.name = node.name
        nodeDef.op = node.opName
        var inputDef = [String]()
        for input in node.inputs {
          switch input {
          case .variable(let id):
            inputDef.append("tensor_\(id)")
          case .node(let id, let idx):
            inputDef.append("\(nodes[id]!.name):\(idx)")
          }
        }
        nodeDef.input = inputDef
        nodeDefs.append(nodeDef)
      }
      graphDef.node = nodeDefs
      var versionDef = Tensorboard_VersionDef()
      versionDef.producer = 22
      graphDef.versions = versionDef
      return graphDef
    }
  }

  /// Add graph for tensorboard graphs dashboard.
  public func addGraph(
    _ value: DynamicGraph,
    wallTime: Double = Date().timeIntervalSince1970
  ) {
    let graphDef = Graph()
    ccv_nnc_dynamic_graph_format(
      value.cGraph, formatGraph, Unmanaged.passUnretained(graphDef).toOpaque())
    graphDef.uniqueName()

    var event = Tensorboard_Event()
    event.graphDef = try! graphDef.proto.serializedData()
    event.wallTime = wallTime
    do {
      try eventLogger.add(event)
    } catch {
      fatalError("Could not add \(event) to log: \(error)")
    }
  }

  /// Add graph for tensorboard graphs dashboard.
  public func addGraph(
    _ value: Model,
    wallTime: Double = Date().timeIntervalSince1970
  ) {
    let graphDef = Graph()
    ccv_cnnp_model_format(value.cModel, formatGraph, Unmanaged.passUnretained(graphDef).toOpaque())
    graphDef.uniqueName()

    var event = Tensorboard_Event()
    event.graphDef = try! graphDef.proto.serializedData()
    event.wallTime = wallTime
    do {
      try eventLogger.add(event)
    } catch {
      fatalError("Could not add \(event) to log: \(error)")
    }
  }
}
