#if os(Linux)
  import XCTest

  XCTMain([
    testCase(MLShapedArrayTests.allTests)
  ])

#endif
