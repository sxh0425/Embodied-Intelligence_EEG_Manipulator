# 第 00 课 · 先看懂代码里的 Python

**这一课不讲机器人，只讲语法。**
以后看代码遇到不懂的符号，回来翻这一课，不用去网上搜。

---

## 1. 点号 `.` 的意思是"里面的"

```python
model.nq      # model 里面的 nq
data.qpos     # data 里面的 qpos
model.jnt_axis[0]   # model 里面的 jnt_axis，再取里面的第 0 个
```

看到 `A.b` 就读成"A 里面的 b"。就这么简单。

---

## 2. 索引 `[0]` 是从 0 开始数的

```python
v = [10, 20, 30]
v[0]   # 10
v[2]   # 30
```

**第一个是 0 不是 1。** 这是初学者最容易错的点。

---

## 3. 函数：`def` 定义，`return` 交答案

```python
def add(a, b):      # def 定义了一个函数，名字叫 add，需要两个东西 a 和 b
    return a + b    # return 是把结果"交出去"
```

warmup 里每一级都是这个结构：

```python
def level4():
    # 这里写你的代码
    return 结果
```

检查程序会拿你 `return` 出来的东西去做对比。**没有 return，或者 return 了 None，就会显示 ⬜ 还没写。**

注意：`return` 后面的代码不会执行，函数碰到 return 就结束了。

---

## 4. 元组：`(a, b, c)` 是"把几个数捆成一包"

```python
return (9, 9, 8)     # 把三个数捆成一包交出去
```

第 1 级要的就是这个。顺序必须对：`(nq, nv, nu)`。

---

## 5. numpy 数组：一排数

```python
data.xpos[3]        # 一个长度 3 的数组，比如 [0.08, 0.0, 0.92]，代表 x, y, z
```

- `len(数组)` → 里面有几个数
- `数组.shape` → 形状，比如 `(12, 3)` 读作"12 行 3 列"
- `数组.reshape(3, 3)` → 把 9 个数重新摆成 3 行 3 列

`shape` 和 `len` 的区别：

```python
len(data.xpos)        # 12   （有 12 个东西）
data.xpos.shape       # (12, 3)  （12 行 3 列）
```

---

## 6. 矩阵乘法 `@`

```python
R @ v      # R 是一个矩阵，v 是一个向量，结果是另一个向量
```

`@` 是给矩阵和向量用的乘法，和普通的 `*` 不一样。第 8 课会用到。

---

## 7. 常用小工具

```python
np.linalg.norm(v)    # v 的长度（模）
np.dot(a, b)         # 两个向量的点积，结果是一个数
np.allclose(a, b)    # 判断 a 和 b 是不是几乎相等，返回 True/False
```

（用之前要先 `import numpy as np`）

---

## 8. `dir()`：不知道一个东西里有什么就用它

```python
import mujoco
print([n for n in dir(model) if not n.startswith("_")])
```

这行会打印出 model 里所有能用的字段名。

**这不是作弊，这是日常干活的方式。** 没人能把几百个字段名背下来。

---

## 9. 注释 `#`

```python
# 井号后面的字是写给人看的，Python 不执行
nq = model.nq   # 也可以写在行尾
```

---

## 自检：你能答出这几个吗

1. `data.qpos` 里的 `data` 是什么？
2. `v = [5, 6, 7]`，`v[1]` 是几？
3. 函数没有 `return` 会怎样？
4. `(12, 3)` 读作什么？

答不出来就再看一遍对应的那一节，然后再去写 warmup。

