# Rust 集合

- **vector** 允许我们一个挨着一个地储存一系列数量可变的值
- **哈希 map**（*hash map*）允许我们将值与一个特定的键（key）相关联

## vector

* vector 允许我们在一个单独的数据结构中储存多个值，所有值在内存中彼此相邻排列
* 使用 & 和 [] 返回一个引用，或者使用 get 方法以索引作为参数来返回一个 Option<&T>

```rust
// 新建
let v: Vec<i32> = Vec::new();
let mut v = vec![1, 2, 3];

// 写入
v.push(4);
v.push(5);

// 读取
let four = &v[3];
let five = v.get(4);

// 遍历
for i in &v {
    println!("{}", i);
}
```

## map

* `HashMap<K, V>` 类型储存了一个键类型 `K` 对应一个值类型 `V` 的映射
* 它通过一个 **哈希函数**（*hashing function*）来实现映射，决定如何将键和值放入内存中
* 像 vector 一样，哈希 map 将它们的数据储存在堆上
* 对于像 `i32` 这样的实现了 `Copy` trait 的类型，其值可以拷贝进哈希 map
* 对于像 `String` 这样拥有所有权的值，其值将被移动而哈希 map 会成为这些值的所有者
* `Entry` 的 `or_insert` 方法在键对应的值存在时就返回这个值的可变引用，如果不存在则将参数作为新值插入并返回新值的可变引用

```rust
// 新建
use std::collections::HashMap;
let mut scores = HashMap::new();

// 写入
scores.insert(String::from("Blue"), 10);
scores.insert(String::from("Yellow"), 50);

// 读取
let score = scores.get("Blue");
println!("{:?}", score);

// 遍历
for (key, value) in &scores {
    println!("{}: {}", key, value);
}
```

