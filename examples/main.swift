import nnc

let df = DataFrame(from: ["/home/liu/workspace/ccv/samples/basmati.png", "/home/liu/workspace/ccv/samples/dex.png", "/home/liu/workspace/ccv/samples/blackbox.png"])
df["image"] = df["main"].toLoadImage()

df["+1"] = df["main", "image"].map { (file: String, image: AnyTensor) -> AnyTensor in
  return Tensor<Float32>(.CPU, .C(1))
}

for i in df["main", "+1"] {
  print(i)
}
