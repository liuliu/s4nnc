#if os(Linux)
import XCTest

XCTMain([
  testCase(NumpyTests.allTests)
])

#endif
