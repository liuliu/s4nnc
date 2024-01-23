import C_nnc
import NNC
import XCTest

final class ModelTests: XCTestCase {

  func testModel() throws {
    let dynamicGraph = DynamicGraph()

    func MulAdd() -> Model {
      let i0 = Input()
      let i1 = Input()
      let i2 = i0 .* i1
      let i3 = Input()
      let i4 = i2 - i3
      return Model([i0, i1, i3], [i4])
    }

    let muladd = MulAdd()
    let tv0 = dynamicGraph.variable(Tensor<Float32>([1.1], .CPU, .C(1)))
    let tv1 = dynamicGraph.variable(Tensor<Float32>([2.2], .CPU, .C(1)))
    let tv2 = dynamicGraph.variable(Tensor<Float32>([0.2], .CPU, .C(1)))
    let tv3 = DynamicGraph.Tensor<Float32>(muladd(inputs: tv0, tv1, tv2)[0])
    XCTAssertEqual(tv3.rawValue[0], 1.1 * 2.2 - 0.2, accuracy: 1e-5)
  }

  func testConvolutionTransposeModel() throws {
    let dynamicGraph = DynamicGraph()

    let convTranspose = ConvolutionTranspose(groups: 1, filters: 2, filterSize: [3, 3])
    let tv0 = dynamicGraph.variable(Tensor<Float32>([1.1], .CPU, .NHWC(1, 2, 2, 3)))
    let tv1 = convTranspose(tv0)
    XCTAssertEqual(tv1.shape[0], 1)
    XCTAssertEqual(tv1.shape[1], 4)
    XCTAssertEqual(tv1.shape[2], 4)
    XCTAssertEqual(tv1.shape[3], 2)
  }

  func testModelBuilder() throws {
    let dynamicGraph = DynamicGraph()

    let builder = ModelBuilder { inputs in
      let i0 = Input()
      let i1 = Input()
      let i2 = i0 .* i1
      return Model([i0, i1], [i2])
    }

    let b0 = dynamicGraph.variable(Tensor<Float32>([1.2], .CPU, .C(1)))
    let b1 = dynamicGraph.constant(Tensor<Float32>([2.2], .CPU, .C(1)))
    let b2 = DynamicGraph.Tensor<Float32>(builder(inputs: b0, b1)[0])
    XCTAssertEqual(b2.rawValue[0], 1.2 * 2.2, accuracy: 1e-5)

    let b3 = dynamicGraph.variable(Tensor<Float32>([1.2, 2.2], .CPU, .C(2)))
    let b4 = dynamicGraph.constant(Tensor<Float32>([2.2, 3.3], .CPU, .C(2)))
    let b5 = DynamicGraph.Tensor<Float32>(builder(inputs: b3, b4)[0])
    XCTAssertEqual(b5.rawValue[0], 1.2 * 2.2, accuracy: 1e-5)
    XCTAssertEqual(b5.rawValue[1], 2.2 * 3.3, accuracy: 1e-5)
  }

  func testSequential() throws {
    let dynamicGraph = DynamicGraph()

    @Sequential
    func MulAdd() -> Model {
      Dense(count: 1)
      ReLU()
    }

    let muladd = MulAdd()
    let tv0 = dynamicGraph.variable(Tensor<Float32>([1.1], .CPU, .C(1)))
    let tv1 = dynamicGraph.variable(Tensor<Float32>([-2.2], .CPU, .C(1)))
    let _ = DynamicGraph.Tensor<Float32>(muladd(inputs: tv0)[0])
    muladd.parameters.clamp(1...1)
    let tv2 = DynamicGraph.Tensor<Float32>(muladd(inputs: tv0)[0])
    let tv3 = DynamicGraph.Tensor<Float32>(muladd(inputs: tv1)[0])
    XCTAssertEqual(tv2.rawValue[0], 2.1, accuracy: 1e-5)
    XCTAssertEqual(tv3.rawValue[0], 0, accuracy: 1e-5)
  }

  func testModelWithScalar() throws {
    let dynamicGraph = DynamicGraph()

    let tv0 = dynamicGraph.variable(Tensor<Float32>([1.1], .CPU, .C(1)))
    let tv1 = dynamicGraph.variable(Tensor<Float32>([2.2], .CPU, .C(1)))

    func MulAdd1() -> Model {
      let i0 = Input()
      let i1 = Input()
      let i2 = i0 .* i1
      let i3 = i2 + 1.2
      return Model([i0, i1], [i3])
    }
    let muladd1 = MulAdd1()
    let tv31 = DynamicGraph.Tensor<Float32>(muladd1(inputs: tv0, tv1)[0])
    XCTAssertEqual(tv31.rawValue[0], 1.1 * 2.2 + 1.2, accuracy: 1e-5)

    func MulAdd2() -> Model {
      let i0 = Input()
      let i1 = Input()
      let i2 = i0 .* i1
      let i3 = i2 - 1.2
      return Model([i0, i1], [i3])
    }
    let muladd2 = MulAdd2()
    let tv32 = DynamicGraph.Tensor<Float32>(muladd2(inputs: tv0, tv1)[0])
    XCTAssertEqual(tv32.rawValue[0], 1.1 * 2.2 - 1.2, accuracy: 1e-5)

    func MulAdd3() -> Model {
      let i0 = Input()
      let i1 = Input()
      let i2 = i0 .* i1
      let i3 = 2.2 + i2
      return Model([i0, i1], [i3])
    }
    let muladd3 = MulAdd3()
    let tv33 = DynamicGraph.Tensor<Float32>(muladd3(inputs: tv0, tv1)[0])
    XCTAssertEqual(tv33.rawValue[0], 1.1 * 2.2 + 2.2, accuracy: 1e-5)

    func MulAdd4() -> Model {
      let i0 = Input()
      let i1 = Input()
      let i2 = i0 .* i1
      let i3 = 1.2 - i2
      return Model([i0, i1], [i3])
    }
    let muladd4 = MulAdd4()
    let tv34 = DynamicGraph.Tensor<Float32>(muladd4(inputs: tv0, tv1)[0])
    XCTAssertEqual(tv34.rawValue[0], 1.2 - 1.1 * 2.2, accuracy: 1e-5)
  }

  func testModelWithParameter() throws {
    let dynamicGraph = DynamicGraph()

    let tv0 = dynamicGraph.variable(Tensor<Float32>([1.1], .CPU, .C(1)))
    let tv1 = dynamicGraph.variable(Tensor<Float32>([2.2], .CPU, .C(1)))

    func MulAdd() -> (Model, Model) {
      let i0 = Input()
      let i1 = Input()
      let i2 = i0 .* i1
      let param = Parameter<Float32>(.CPU, .C(1))
      let i3 = i2 + param
      return (param, Model([i0, i1], [i3]))
    }
    let (param, muladd) = MulAdd()
    muladd.compile(inputs: tv0, tv1)
    param.weight.copy(from: Tensor<Float32>([3.1], .CPU, .C(1)))
    let tv3 = DynamicGraph.Tensor<Float32>(muladd(inputs: tv0, tv1)[0])
    XCTAssertEqual(tv3.rawValue[0], 1.1 * 2.2 + 3.1, accuracy: 1e-5)
  }

  func testModelDiv() throws {
    let dynamicGraph = DynamicGraph()

    func DivRec() -> Model {
      let i0 = Input()
      let i1 = Input()
      let i2 = i0 ./ i1
      let i3 = Input()
      let i4 = 0.5 / i3
      return Model([i0, i1, i3], [i2, i4])
    }

    let div = DivRec()
    let tv0 = dynamicGraph.variable(Tensor<Float32>([1.1], .CPU, .C(1)))
    let tv1 = dynamicGraph.variable(Tensor<Float32>([2.2], .CPU, .C(1)))
    let tv2 = dynamicGraph.variable(Tensor<Float32>([0.2], .CPU, .C(1)))
    let tv3s = div(inputs: tv0, tv1, tv2).map { $0.as(of: Float32.self) }
    XCTAssertEqual(tv3s[0].rawValue[0], 1.1 / 2.2, accuracy: 1e-5)
    XCTAssertEqual(tv3s[1].rawValue[0], 0.5 / 0.2, accuracy: 1e-5)
  }

  func testModelScaledDotProductAttention() throws {
    let dynamicGraph = DynamicGraph()
    let q = dynamicGraph.variable(Tensor<Float32>(.CPU, .NHWC(1, 10, 8, 20)))
    let k = dynamicGraph.variable(Tensor<Float32>(.CPU, .NHWC(1, 20, 8, 20)))
    let v = dynamicGraph.variable(Tensor<Float32>(.CPU, .NHWC(1, 20, 8, 30)))
    q.randn()
    k.randn()
    v.randn()
    let scaledDotProductAttention = ScaledDotProductAttention(scale: 1)
    let out = scaledDotProductAttention(queries: q, keys: k, values: v)
    var dot = Functional.matmul(
      left: q.transposed(1, 2), right: k.transposed(1, 2), rightTranspose: (2, 3))
    dot = dot.reshaped(.NC(8 * 10, 20))
    dot.softmax()
    dot = dot.reshaped(.NHWC(1, 8, 10, 20))
    let out2 = (dot * v.transposed(1, 2)).transposed(1, 2)
    for i in 0..<10 {
      for j in 0..<8 {
        for k in 0..<30 {
          XCTAssertEqual(out[0, i, j, k], out2[0, i, j, k], accuracy: 1e-5)
        }
      }
    }
  }

  func testCustomModel() throws {
    let dynamicGraph = DynamicGraph()
    var params = ccv_nnc_tensor_param_t()
    params.type = Int32(CCV_TENSOR_CPU_MEMORY)
    params.datatype = Int32(CCV_32F)
    params.format = Int32(CCV_TENSOR_FORMAT_NHWC)
    params.dim = (80, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    var cmdParams = CmdParamsFactory.factory.newParams()
    cmdParams.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    cmdParams.blas.a = (1.0 / 80, 0, 0)
    let initFirstWeightState = ccv_cnnp_cmd_exec_io_set_by(
      ccv_nnc_cmd(CCV_NNC_RANDOM_NORMAL_FORWARD, nil, cmdParams, 0), ccv_nnc_hint_t(), 0, params)
    params.dim = (80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    cmdParams.blas.a = (0, 0, 0)
    let initFirstBiasState = ccv_cnnp_cmd_exec_io_set_by(
      ccv_nnc_cmd(CCV_NNC_SET_FORWARD, nil, cmdParams, 0), ccv_nnc_hint_t(), 0, params)
    params.dim = (20, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    cmdParams.blas.a = (1.0 / 20, 0, 0)
    let initSecondWeightState = ccv_cnnp_cmd_exec_io_set_by(
      ccv_nnc_cmd(CCV_NNC_RANDOM_NORMAL_FORWARD, nil, cmdParams, 0), ccv_nnc_hint_t(), 0, params)
    params.dim = (20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    cmdParams.blas.a = (0, 0, 0)
    let initSecondBiasState = ccv_cnnp_cmd_exec_io_set_by(
      ccv_nnc_cmd(CCV_NNC_SET_FORWARD, nil, cmdParams, 0), ccv_nnc_hint_t(), 0, params)
    let feedForward = CustomModel(
      inputs: [
        .inputOrOutput, .sharedTensorAsTrainable(initFirstWeightState),
        .sharedTensorAsTrainable(initFirstBiasState),
        .sharedTensorAsTrainable(initFirstWeightState),
        .sharedTensorAsTrainable(initFirstBiasState),
        .sharedTensorAsTrainable(initSecondWeightState),
        .sharedTensorAsTrainable(initSecondBiasState),
      ], outputs: [.inputOrOutput], hint: Hint(),
      shapeInference: {
        (
          _: ccv_nnc_cmd_t, inputs: UnsafePointer<ccv_nnc_tensor_param_t>?, _: Int32,
          _: ccv_nnc_hint_t, outputs: UnsafeMutablePointer<ccv_nnc_tensor_param_t>?, _: Int32
        ) -> Void in
        var params = inputs![0]
        params.dim.1 = 20
        outputs![0] = params
      },
      execute: {
        (
          cmd: ccv_nnc_cmd_t, _: ccv_nnc_hint_t, _: Int32,
          inputs: UnsafePointer<UnsafeMutablePointer<ccv_nnc_tensor_t>?>?, inputSize: Int32,
          outputs: UnsafePointer<UnsafeMutablePointer<ccv_nnc_tensor_t>?>?, outputSize: Int32,
          streamContext: OpaquePointer?
        ) -> Int32 in
        if cmd.cmd == UInt32(CCV_NNC_CUSTOM_FORWARD) {
          var denseParams = CmdParamsFactory.factory.newParams()
          denseParams.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
          denseParams.blas.transpose_b = (0, 1)
          let gemm = ccv_nnc_cmd(CCV_NNC_GEMM_FORWARD, nil, denseParams, 0)
          var tensorParams = inputs![0]!.pointee.info
          tensorParams.dim = (tensorParams.dim.0, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
          var fc10 = ccv_nnc_tensor_new(nil, tensorParams, 0)
          var fc11 = ccv_nnc_tensor_new(nil, tensorParams, 0)
          ccv_nnc_cmd_exec(gemm, ccv_nnc_hint_t(), 0, inputs, 3, &fc10, 1, streamContext)
          ccv_nnc_cmd_exec(
            gemm, ccv_nnc_hint_t(), 0, [inputs![0], inputs![3], inputs![4]], 3, &fc11, 1,
            streamContext)
          var geluParams = CmdParamsFactory.factory.newParams()
          geluParams.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
          geluParams.gelu.tanh = 0
          let gelu = ccv_nnc_cmd(CCV_NNC_GELU_FORWARD, nil, geluParams, 0)
          ccv_nnc_cmd_exec(gelu, ccv_nnc_hint_t(), 0, &fc11, 1, &fc11, 1, streamContext)
          var mulParams = CmdParamsFactory.factory.newParams()
          mulParams.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
          mulParams.blas.a = (1, 0, 0)
          let mul = ccv_nnc_cmd(CCV_NNC_MUL_FORWARD, nil, mulParams, 0)
          ccv_nnc_cmd_exec(mul, ccv_nnc_hint_t(), 0, [fc10, fc11], 2, &fc10, 1, streamContext)
          ccv_nnc_tensor_free(fc11)
          ccv_nnc_cmd_exec(
            gemm, ccv_nnc_hint_t(), 0, [fc10, inputs![5], inputs![6]], 3, outputs, 1, streamContext)
          ccv_nnc_tensor_free(fc10)
        } else if cmd.cmd == UInt32(CCV_NNC_CUSTOM_BACKWARD) {
          var denseParams = CmdParamsFactory.factory.newParams()
          denseParams.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
          denseParams.blas.transpose_b = (0, 1)
          let gemm = ccv_nnc_cmd(CCV_NNC_GEMM_FORWARD, nil, denseParams, 0)
          var tensorParams = inputs![1]!.pointee.info
          tensorParams.dim = (tensorParams.dim.0, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
          var fc10 = ccv_nnc_tensor_new(nil, tensorParams, 0)
          var fc11 = ccv_nnc_tensor_new(nil, tensorParams, 0)
          ccv_nnc_cmd_exec(gemm, ccv_nnc_hint_t(), 0, inputs! + 1, 3, &fc10, 1, streamContext)
          ccv_nnc_cmd_exec(
            gemm, ccv_nnc_hint_t(), 0, [inputs![1], inputs![4], inputs![5]], 3, &fc11, 1,
            streamContext)
          var geluParams = CmdParamsFactory.factory.newParams()
          geluParams.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
          geluParams.gelu.tanh = 0
          let gelu = ccv_nnc_cmd(CCV_NNC_GELU_FORWARD, nil, geluParams, 0)
          var fc12 = ccv_nnc_tensor_new(nil, tensorParams, 0)
          ccv_nnc_cmd_exec(gelu, ccv_nnc_hint_t(), 0, &fc11, 1, &fc12, 1, streamContext)
          var mulParams = CmdParamsFactory.factory.newParams()
          mulParams.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
          mulParams.blas.a = (1, 0, 0)
          var fc2 = ccv_nnc_tensor_new(nil, tensorParams, 0)
          let gemmBack = ccv_nnc_cmd(CCV_NNC_GEMM_BACKWARD, nil, denseParams, 0)
          ccv_nnc_cmd_exec(
            gemmBack, ccv_nnc_hint_t(), 0, [inputs![0], nil, inputs![6], inputs![7]], 4, &fc2, 1,
            streamContext)
          var fc21 = ccv_nnc_tensor_new(nil, tensorParams, 0)
          let mul = ccv_nnc_cmd(CCV_NNC_MUL_FORWARD, nil, mulParams, 0)
          ccv_nnc_cmd_exec(mul, ccv_nnc_hint_t(), 0, [fc10, fc2], 2, &fc21, 1, streamContext)
          ccv_nnc_cmd_exec(mul, ccv_nnc_hint_t(), 0, [fc2, fc12], 2, &fc10, 1, streamContext)
          let geluBack = ccv_nnc_cmd(CCV_NNC_GELU_BACKWARD, nil, geluParams, 0)
          ccv_nnc_cmd_exec(geluBack, ccv_nnc_hint_t(), 0, [fc21, fc11], 2, &fc12, 1, streamContext)
          ccv_nnc_tensor_free(fc11)
          ccv_nnc_tensor_free(fc21)
          tensorParams = inputs![1]!.pointee.info
          var din0 = ccv_nnc_tensor_new(nil, tensorParams, 0)
          ccv_nnc_cmd_exec(
            gemmBack, ccv_nnc_hint_t(), 0, [fc10, nil, inputs![2], inputs![3]], 4, &din0, 1,
            streamContext)
          ccv_nnc_tensor_free(fc10)
          var din1 = ccv_nnc_tensor_new(nil, tensorParams, 0)
          ccv_nnc_cmd_exec(
            gemmBack, ccv_nnc_hint_t(), 0, [fc12, nil, inputs![4], inputs![5]], 4, &din1, 1,
            streamContext)
          ccv_nnc_tensor_free(fc12)
          var sumParams = CmdParamsFactory.factory.newParams()
          sumParams.size.dim = (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
          let sum = ccv_nnc_cmd(CCV_NNC_EWSUM_FORWARD, nil, sumParams, 0)
          ccv_nnc_cmd_exec(sum, ccv_nnc_hint_t(), 0, [din0, din1], 2, outputs, 1, streamContext)
          ccv_nnc_tensor_free(din0)
          ccv_nnc_tensor_free(din1)
        }
        return Int32(CCV_NNC_EXEC_SUCCESS)
      })
    let input = dynamicGraph.variable(Tensor<Float32>(.CPU, .NC(4, 20)))
    input.randn()
    let output = feedForward(inputs: input)[0].as(of: Float32.self)

    let (fc10, fc11, fc2, feedForward2) =
      ({ () -> (Model, Model, Model, Model) in
        let x = Input()
        let fc10 = Dense(count: 80)
        let fc11 = Dense(count: 80)
        var out = fc10(x)
        out = out .* GELU()(fc11(x))
        let fc2 = Dense(count: 20)
        out = fc2(out)
        return (fc10, fc11, fc2, Model([x], [out]))
      })()
    feedForward2.compile(inputs: input)
    let tensor = Tensor<Float32>(.CPU, .NC(80, 20))
    feedForward.parameters(for: .index(0)).copy(to: tensor)
    fc10.weight.copy(from: tensor)
    feedForward.parameters(for: .index(2)).copy(to: tensor)
    fc11.weight.copy(from: tensor)
    let tensor2 = Tensor<Float32>(.CPU, .NC(20, 80))
    feedForward.parameters(for: .index(4)).copy(to: tensor2)
    fc2.weight.copy(from: tensor2)
    let output2 = feedForward2(inputs: input)[0].as(of: Float32.self)
    for i in 0..<4 {
      for j in 0..<20 {
        XCTAssertEqual(output[i, j], output2[i, j], accuracy: 1e-5)
      }
    }
    let grad = dynamicGraph.variable(.CPU, .NC(4, 20), of: Float32.self)
    grad.randn()
    output2.grad = grad
    input.requiresGrad = true
    output2.backward(to: input)
    let outGrad2 = input.grad!.as(of: Float32.self)
    output.grad = grad
    input.grad = nil
    output.backward(to: input)
    let outGrad = input.grad!.as(of: Float32.self)
    for i in 0..<4 {
      for j in 0..<20 {
        XCTAssertEqual(outGrad[i, j], outGrad2[i, j], accuracy: 1e-5)
      }
    }
  }

  static let allTests = [
    ("testModel", testModel),
    ("testConvolutionTransposeModel", testConvolutionTransposeModel),
    ("testModelBuilder", testModelBuilder),
    ("testSequential", testSequential),
    ("testModelWithScalar", testModelWithScalar),
    ("testModelWithParameter", testModelWithParameter),
    ("testModelDiv", testModelDiv),
    ("testModelScaledDotProductAttention", testModelScaledDotProductAttention),
    ("testCustomModel", testCustomModel),
  ]
}
