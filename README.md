[Documents](https://www.yuque.com/jamily.blog/rmnn)

# Why I share this repository

Remember to write zmTools this library was originally for their convenience, speed up their code writing speed. Abstract out a lot of common code, for the future to add and modify the basis, can quickly iteration experiment, verify their own ideas right or wrong.

There are already two libraries, and the existing work is based on them. The first library is the [ZMTools](https://github.com/lizhemin15/zmtools) core library (github recently had problems accessing it, the latest one is posted on Gitee, renamed [RMNN](https://gitee.com/lizhemin15/rmnn), Represent Matrix with Neural Network), the most important stuff is here. The second library, [AIR-Net](https://github.com/lizhemin15/AIR-Net), is based on ZMTools and mainly adds main.ipynb code at the application level, which is convenient to quickly repeat the previous article:

[1] [DMF+TV](https://arxiv.org/abs/2007.14581#:~:text=Regularized%20Deep%20Matrix%20Factorized%20%28RDMF%29%20model%20for%20image,variation.%20We%20demonstrate%20the%20effectiveness%20of%20our%20RDMF)

[2] [AIR-Net](https://arxiv.org/abs/2110.07557v3)


Future articles and codebase that use this core codebase will be placed here.

In line with the convenience of their own review and others to use the principle, here will be the library's design ideas, organizational structure, the use of the way to elaborate.

# Design idea

1. Idea of decoupling and modularization

Decouple functions, data from code, and core algorithms within code from input and output. The advantage of decoupling is that the code can grow by iterating through the parts rather than pulling them together. The specific expression of decoupling is code modularization, which is very common in the field of electronic product design. It can be traced back to von Neumann's system, in which each part of the computer system performs its own duties and the interaction interface between standardization. This step design is of special significance to the subsequent prosperity and development of the computer.

2. Multi-scale aggregation idea

Multi-scale thought in mathematics and computer are many, in mathematics is a typical pyramid algorithm, wavelet, the computer inside a typical example is the abstraction layers of the programming language, from the controlling hardware level, and then to subsequent high-level language, step by step is let's operating flexibility to reduce at the same time, improve our use convenience.

Of course, many high-level languages strive for convenience without losing flexibility. I'm not at that level yet, so what I can do is break down the code hierarchy more clearly, showing a layer of abstraction that I can use later to decide which level of code to change.





