# Rust 基础

## 变量

* 变量默认是不变的，需要改变可以用 mut
* 使用相同的变量名并重复使用 let 关键字来遮蔽变量 [ 不用再去费力地想另一个名字了 ]
* 通过使用 let，我们可以对一个值进行一些转换，但在这些转换完成后，变量将是不可变的
* mut 和遮蔽之间的另一个区别是，因为我们在再次使用 let 关键字时有效地创建了一个新的变量，所以我们可以改变值的类型

## 动态数组

```rust
let v: Vec = Vec::new();
let v = vec![1, 2, 3];

v.push(5);

println!("{:?}", v[0]);
```

## 哈希表

```rust
use std::collections::HashMap;
let mut scores = HashMap::new();

scores.insert(String::from("Blue"), 10); 
scores.insert(String::from("Yellow"), 50);
```

## 元组

> 元组是一个固定（元素）长度的列表，每个元素类型可以不一样
> 元组在 Rust 中很有用，因为它可以用于函数的返回值，相当于把多个想返回的值捆绑在一起，一次性返回

```rust
let x: (i32, f64, u8) = (500, 6.4, 1);

let five_hundred = x.0;
let six_point_four = x.1;
let one = x.2;
```

## 结构体

```rust
struct User {
  active: bool, 
  username: String, 
  email: String, 
  age: u64,
}

let user1 = User { 
  active: true, 
  username: String::from("someusername123"), 
  email: String::from("someone@example.com"), 
  age: 1, 
};
```

## 枚举

> 学术上，通常把枚举叫作和类型（sum type），把结构体叫作积类型（product type）

```rust
enum IpAddrKind { 
  V4, 
  V6,
}

let four = IpAddrKind::V4;
let six = IpAddrKind::V6;
```

## 函数

* 语句（statement）是执行一些操作但不返回值的指令。表达式（expression）计算并产生一个值
* 函数有返回值 return 语句不能有分号
* 代码块结尾最后一句不加分号，表示把值返回回去

## 闭包

> 闭包是另一种风格的函数。它使用两个竖线符号 || 定义，而不是用 fn () 来定义

```rust
// 标准的函数定义
fn  add_one_v1   (x: u32) -> u32 { x + 1 }
// 闭包的定义，请注意形式对比
let add_one_v2 = |x: u32| -> u32 { x + 1 };
// 闭包的定义2，省略了类型标注
let add_one_v3 = |x|             { x + 1 };
// 闭包的定义3，花括号也省略了
let add_one_v4 = |x|               x + 1  ;
```
