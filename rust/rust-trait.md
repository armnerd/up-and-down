# Rust 泛型、trait 和生命周期

* 每一个编程语言都有高效处理重复概念的工具。在 Rust 中其工具之一就是 泛型（generics）
* trait是一个定义泛型行为的方法。trait 可以与泛型结合来将泛型限制为拥有特定行为的类型，而不是任意类型
* **生命周期**（*lifetimes*），它是一类允许我们向编译器提供引用如何相互关联的泛型。Rust 的生命周期功能允许在很多场景下借用值的同时仍然使编译器能够检查这些引用的有效性

## trait

* trait 告诉 Rust 编译器某个特定类型拥有可能与其他类型共享的功能
* trait 类似于其他语言中常被称为 接口（interfaces）的功能，虽然有一些不同
* trait 定义是一种将方法签名组合起来的方法，目的是定义一个实现某些目的所必需的行为的集合
* 有时为 trait 中的某些或全部方法提供默认的行为，而不是在每个类型的每个实现中都定义自己的行为是很有用的。这样当为某个特定类型实现 trait 时，可以选择保留或重载每个方法的默认行为。