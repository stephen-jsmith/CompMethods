# Computational Methods - Fall 2025

Welcome to the repository for **Computational Methods** coursework.

## Overview

This repository contains assignment and project work for the Fall 2025 Computational Methods class.

## Structure

- `objects/` — Anything that is an object. For assignment 2, this includes Nodes and Members
- `scripts/` — Where functions live that are not a part of the objects.


## Getting Started

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/CompMethods.git
    ```
2. Navigate to the project directory:
    ```bash
    cd CompMethods
    ```

## Requirements

- Python 3.x
-- SymPy (member matrices return `sympy.Matrix`) — install with `pip install -r requirements.txt`

Note: Member objects now return symbolic `sympy.Matrix` instances for stiffness
and transformation matrices. If you need numeric assembly, convert or evaluate
these matrices to numeric numpy arrays using `.evalf(subs=...)` and `numpy.array(...)`.
