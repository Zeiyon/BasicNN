# Basic_XOR.py

A minimal scratchpad demonstrating the XOR setup and variables for a two-layer neural network.  
This file is mainly for understanding the structure and math and is **not used in the main run**.

---

## Basic Structure

![XOR Neural Network Input/Output Math Simplification](https://i.gyazo.com/88d631b9500b39966b736020fe29932b.png)

The dataset, **Inputs**, is made up of 4 points on the Cartesian plane, each containing 2 values: an x and a y coordinate. Plotting these 4 points shows that they form a square.

The output **Y** (not to be confused with the y from the Cartesian plane) is set in the same order as the Inputs, where the *n-th* element is the correct output value for the *n-th* input.

Keep in mind, it does not need to be done this way. This is just how the input/output system is designed here — any consistent way of keeping track of inputs and outputs is fine.

Next, we convert the inputs and outputs into column vectors. Each input \((x_1, x_2)\) becomes two column vectors, \(x_1\) and \(x_2\), which can then be combined into a matrix. Since the output is only made up of one value per input, it can be represented as a single column vector \(y\).

---

## Neural Network Architecture

![XOR Neural Network Architecture](https://i.gyazo.com/6a968fd93769b807c47227644084cb5c.png)

Above is the architecture of the neural network. It consists of:
- 2 inputs: \(x_1\) and \(x_2\)
- A hidden layer with 2 neurons: \(h_1\) and \(h_2\)
- A single output neuron

Below the diagram is the math for \(H_1\) and \(H_2\). The output is computed in the same way:

Output = H1·W31 + H2·W32 + b3

---

## Matrix Simplification

![XOR Neural Network Math Simplification](https://i.gyazo.com/31dc177384430912b6599dac0beaf5cd.png)

Finally, the math can be simplified and written in matrix form. We already have the intuition for how the network works, but with more complex architectures, it becomes important to understand the linear algebra representation. This allows us to see what is happening at each layer without getting caught up in dozens or hundreds of individual equations.

---

### Setup
1) Optional: create a virtual environment  
```bash
python3 -m venv venv && source venv/bin/activate
```

2) Install dependencies  
```bash
pip install numpy matplotlib
```

3) Execute the main script to train and visualize:
```bash
python3 Basic_XOR.py
```