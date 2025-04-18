# Rust 字符串

Rust 的核心语言中只有一种字符串类型：`str`，字符串 slice，它通常以被借用的形式出现，`&str`。

`String` 是一个 `Vec<u8>` 的封装

```rust
// 创建
let mut s = String::new();
let data = "initial contents";
let s = data.to_string();
let s = String::from("initial contents");
```