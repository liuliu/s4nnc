import nnc

let dataframe = DataFrame(from: [1, 2, 3, 4])
let iter = dataframe["a"]
iter.prefetch(2)
for i in iter {
  print(i)
}
