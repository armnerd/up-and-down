# Rust 结构体与枚举

> 结构体不是创建自定义类型的唯一方法

## 结构体

```rust
struct User {
    active: bool,
    username: String,
    email: String,
    sign_in_count: u64,
}

struct Color(i32, i32, i32);
struct Point(i32, i32, i32);

fn main() {
    let black = Color(0, 0, 0);
    let origin = Point(0, 0, 0);
}
```

* 结构体和元组一样，结构体的每一部分可以是不同类型
* 结构体比元组更灵活：不需要依赖顺序来指定或访问实例中的值
* Rust 并不允许只将某个字段标记为可变
* 元组结构体有着结构体名称提供的含义，但没有具体的字段名，只有字段的类型

## 方法

```rust
#[derive(Debug)]
struct Rectangle {
    width: u32,
    height: u32,
}

impl Rectangle {
    fn area(&self) -> u32 {
        self.width * self.height
    }
}

fn main() {
    let rect1 = Rectangle {
        width: 30,
        height: 50,
    };

    println!(
        "The area of the rectangle is {} square pixels.",
        rect1.area()
    );
}
```

* 方法 与函数类似：它们使用 fn 关键字和名称声明，可以拥有参数和返回值，并且它们第一个参数总是 self，它代表调用该方法的结构体实例
* 这里选择 `&self` 的理由跟在函数版本中使用 `&Rectangle` 是相同的：我们并不想获取所有权，只希望能够读取结构体中的数据，而不是写入
* 如果想要在方法中改变调用方法的实例，需要将第一个参数改为 `&mut self`
* 给出接收者和方法名的前提下，Rust 可以明确地计算出方法是仅仅读取（&self），做出修改（&mut self）或者是获取所有权（self）

## 关联函数

```rust
impl Rectangle {
    fn square(size: u32) -> Rectangle {
        Rectangle {
            width: size,
            height: size,
        }
    }
}

let sq = Rectangle::square(3);
```

* 所有在 impl 块中定义的函数被称为关联函数 `associated function`，因为它们与 impl 后面命名的类型相关
* 我们可以定义不以 self 为第一参数的关联函数（因此不是方法），因为它们并不作用于一个结构体的实例
* 关联函数经常被用作返回一个结构体新实例的构造函数
* 使用结构体名和 `::` 语法来调用这个关联函数
* 每个结构体都允许拥有多个 impl 块

## 枚举

```rust
enum IpAddr {
    V4(u8, u8, u8, u8),
    V6(String),
}

let home = IpAddr::V4(127, 0, 0, 1);
let loopback = IpAddr::V6(String::from("::1"));
```

* 枚举 `enumerations`允许你通过列举可能的 成员 `variants` 来定义一个类型
* 用枚举替代结构体还有另一个优势：每个成员可以处理不同类型和数量的数据
* 枚举和结构体还有另一个相似点：就像可以使用 impl 来为结构体定义方法那样，也可以在枚举上定义方法

## Option

```rust
enum Option<T> {
    Some(T),
    None,
}
```

* Option 是标准库定义的另一个枚举。Option 类型应用广泛是因为它编码了一个非常普遍的场景，即一个值要么有值要么没值
* Rust 并没有很多其他语言中有的空值功能。空值（Null ）是一个值，它代表没有值。在有空值的语言中，变量总是这两种状态之一：空值和非空值
* 空值的问题在于当你尝试像一个非空值那样使用一个空值，会出现某种形式的错误

## match

```rust
enum Coin {
    Penny,
    Nickel,
    Dime,
    Quarter,
}

fn value_in_cents(coin: Coin) -> u8 {
    match coin {
        Coin::Penny => {
            println!("Lucky penny!");
            1
        }
        Coin::Nickel => 5,
        Coin::Dime => 10,
        Coin::Quarter => 25,
    }
}

fn plus_one(x: Option<i32>) -> Option<i32> {
    match x {
        None => None,
        Some(i) => Some(i + 1),
    }
}

match dice_roll {
    3 => add_fancy_hat(),
    7 => remove_fancy_hat(),
    other => move_player(other),
}

match dice_roll {
    3 => add_fancy_hat(),
    7 => remove_fancy_hat(),
    _ => reroll(),
}

match dice_roll {
    3 => add_fancy_hat(),
    7 => remove_fancy_hat(),
    _ => (),
}
```

* Rust 有一个叫做 match 的极为强大的控制流运算符，它允许我们将一个值与一系列的模式相比较，并根据相匹配的模式执行相应代码
* match 的力量来源于模式的表现力以及编译器检查，它确保了所有可能的情况都得到处理
* 可以认为 if let 是 match 的一个语法糖，它当值匹配某一模式时执行代码而忽略所有其他值

```rust
let mut count = 0;
if let Coin::Quarter(state) = coin {
    println!("State quarter from {:?}!", state);
} else {
    count += 1;
}
```