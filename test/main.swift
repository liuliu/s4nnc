#if os(Linux)
import XCTest

XCTMain([
  testCase(DataFrameTests.allTests),
  testCase(GraphTests.allTests),
  testCase(LossTests.allTests),
  testCase(ModelTests.allTests),
  testCase(OptimizerTests.allTests),
  testCase(StoreTests.allTests)
])

#endif
