#if os(Linux)
import XCTest

XCTMain([
  testCase(ModelTests.allTests),
  testCase(GraphTests.allTests)
])

#endif
