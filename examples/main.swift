import nnc

let df = DataFrame(from: [1, 2, 3, 4, 5])

df["+1"] = df["main"].map { (i: Int) -> Int in
  i + 1
}

df["2"] = .from([1, 1, 1, 1, 1])

df["++"] = df["main", "+1"].map { (i: Int, j: Int) -> Int in
  i + j
}
df["3"] = .from([1, 1, 1, 1, 1])
df["4"] = .from([1, 1, 1, 1, 1])
df["5"] = .from([1, 1, 1, 1, 1])
df["6"] = .from([1, 1, 1, 1, 1])
df["9"] = .from([1, 1, 1, 1, 1])
df["10"] = .from([1, 1, 1, 1, 1])
df["z"] = df["main", "+1", "++", "2", "3", "4", "5", "6", "9", "10"].map { (c0: Int, c1: Int, c2: Int, c3: Int, c4: Int, c5: Int, c6: Int, c7: Int, c8: Int, c9: Int) -> Int in
  return c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9
}

for i in df["main", "+1", "++", "z"] {
  print(i)
}
