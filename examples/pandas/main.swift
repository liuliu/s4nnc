import nnc
import PythonKit

let sys = Python.import("sys")
let result: PythonObject = (sys.version_info.major == 3)
if result == true {
  print(sys.version_info)
}
