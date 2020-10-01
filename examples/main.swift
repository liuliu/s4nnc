import nnc

let dataframe = DataFrame(from: [1, 2, 3, 4])
let iter = dataframe["main", Int.self]

dataframe["1"] = .from(10)
dataframe["2"] = .from([4, 3, 2, 1])

dataframe["rename"] = dataframe["2"]

iter.prefetch(2)

for i in iter {
  print(i)
}

for i in iter {
  print(i)
}

let iter2 = dataframe["1"]
for i in iter2 {
  print(i)
}

for i in dataframe["2"] {
  print(i)
}

for i in dataframe["main", "rename", "1"] {
  print(i)
}
