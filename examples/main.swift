import nnc

let df = DataFrame(fromCSV: "/home/liu/workspace/paratext/tests/hepatitis-chinese.csv")!

let columns = df.columns
for i in df[columns] {
  print(i as! [String])
}

/*
let df = DataFrame(from: [1, 2])
var tensor = Tensor<Float32>(.CPU, .C(1))
tensor[0] = 1.2
df["image"] = .from([tensor, tensor])
for i in df["0", "image"] {
  print((i[1] as! Tensor<Float32>)[0])
}

let df = DataFrame(from: ["/home/liu/workspace/ccv/samples/basmati.png", "/home/liu/workspace/ccv/samples/dex.png", "/home/liu/workspace/ccv/samples/blackbox.png"])
df["image"] = df["0"].toLoadImage()

df["+1"] = df["0", "image"].map { (file: String, image: AnyTensor) -> AnyTensor in
  return Tensor<Float32>(.CPU, .C(1))
}

for i in df["0", "+1"] {
  print(i)
}
*/
