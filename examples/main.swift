import nnc

let dataframe = DataFrame(from: ["/home/liu/workspace/ccv/samples/chessbox.png", "/home/liu/workspace/ccv/samples/dex.png"])
dataframe["image"] = dataframe["main"].toLoadImage()
for i in dataframe["main", "image"] {
  print(i)
}

enum MyStruct {
  case value(Float32)
  case string(String)
}

let df = DataFrame(from: [MyStruct.value(1.0), nil, MyStruct.string("theirs")])

for i in df["main"] {
  print(i)
}
