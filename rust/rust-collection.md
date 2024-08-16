# Rust 集合

- **vector** 允许我们一个挨着一个地储存一系列数量可变的值
- **字符串**（*string*）是字符的集合。我们之前见过 `String` 类型，不过在本章我们将深入了解。
- **哈希 map**（*hash map*）允许我们将值与一个特定的键（key）相关联。这是一个叫做 *map* 的更通用的数据结构的特定实现。

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

