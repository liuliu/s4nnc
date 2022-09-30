import NNC
import PythonKit

let sys = Python.import("sys")
let result: PythonObject = (sys.version_info.major == 3)
if result == true {
  print(sys.version_info)
}

let gc = Python.import("gc")

let y = PythonObject(10)
let lambda1 = PythonFunction { x in x * y }
let lambda2 = PythonFunction { (x: PythonObject) -> PythonConvertible in x + y }
print(Python.list(Python.map(lambda1, [10, 12, 14])))
print(Python.list(Python.map(lambda1, [2, 3, 4])))
print(Python.list(Python.map(lambda2, [2, 3, 4])))
print(gc.get_stats())
