#if os(Linux)
import XCTest

XCTMain([
  testCase(DataFrameTests.allTests),
  testCase(GraphTests.allTests),
  testCase(ModelTests.allTests),
  testCase(OptimizerTests.allTests)
])

#endif
