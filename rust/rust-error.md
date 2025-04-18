# Rust 错误

## panic

> 与不可恢复的错误

当出现 panic 时，程序默认会开始 **展开**（*unwinding*），这意味着 Rust 会回溯栈并清理它遇到的每一个函数的数据，不过这个回溯并清理的过程有很多工作。另一种选择是直接 **终止**（*abort*），这会不清理数据就退出程序。那么程序所使用的内存需要由操作系统来清理。

```rust
fn main() {
    panic!("crash and burn");
}
```

## Result

* 大部分错误并没有严重到需要程序完全停止执行。有时，一个函数会因为一个容易理解并做出反应的原因失败
* T 代表成功时返回的 Ok 成员中的数据的类型，而 E 代表失败时返回的 Err 成员中的错误的类型

```rust
enum Result<T, E> {
    Ok(T),
    Err(E),
}

let f = File::open("hello.txt");
let f = match f {
    Ok(file) => file,
    Err(error) => {
        panic!("Problem opening the file: {:?}", error)
    },
};
```