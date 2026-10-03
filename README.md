# Solutions to Karatzas and Shreve, *Brownian Motion and Stochastic Calculus* (Springer, 2nd ed., 1991)

Authors: Zijun Meng and Claude Opus 5.5

## Contents

The book (Graduate Texts in Mathematics 113) has 293 numbered Problems and Exercises in
its six chapters. At the end of each chapter, a section "Solutions to Selected Problems"
solves 100 of them; for 15 of those it prints only a hint, a sketch, or some of the parts.
This document solves the other 193 completely, and for the 15 it supplies what the book
leaves out: 208 solutions in all.

| Chapter | Problems | Exercises | solved in the book | solved there only in part | no solution in the book | solved here |
|---|---:|---:|---:|---:|---:|---:|
| 1. Martingales, Stopping Times, and Filtrations | 43 | 8 | 20 | 2 | 29 | **31** |
| 2. Brownian Motion | 44 | 18 | 22 | 5 | 35 | **40** |
| 3. Stochastic Integration | 40 | 33 | 15 | 5 | 53 | **58** |
| 4. Brownian Motion and Partial Differential Equations | 6 | 14 | 4 | 1 | 15 | **16** |
| 5. Stochastic Differential Equations | 30 | 36 | 18 | 1 | 47 | **48** |
| 6. P. Lévy's Theory of Brownian Local Time | 11 | 10 | 6 | 1 | 14 | **15** |
| **Total** | 174 | 119 | 85 | 15 | 193 | **208** |

Each chapter of the document opens with the list of the items it covers, the items the
book solves only in part, and those the book solves in full (which are not repeated).

**The statements are not reproduced.** Each solution is headed only by the book's own word
for the item (Problem or Exercise) and its number, so read the statement in the book first.
Notation follows the book. As in the book, a result is cited by two numbers inside its own
chapter ("Theorem 3.8", "(2.7)") and by three from another chapter ("Theorem 1.3.8").
Remarks inside the solutions point out the places where a statement of the book contains a
misprint or needs an extra hypothesis; for example, the factor $(r-\lVert x\rVert^2)$ in
(4.2.24)–(4.2.25) should be $(r^2-\lVert x\rVert^2)$; Exercise 3.7.7 needs
$\int h\,d\mu<\infty$, since a $\sigma$-finite measure need not be finite on compact sets;
and the clause about omitting (4.13) in Theorem 6.4.3 refers to (4.14).

| File | Contents |
|---|---|
| `KaratzasShreveSolutions.tex` / `.pdf` | the solutions in one self-contained document (solutions only, without the statements) |
| `code/` | standalone numerical checks, one script per item that asks for an explicit formula (see below) |

## How the solutions were checked

Every solution was written by one agent and then examined by two others working
independently: a referee who re-derived the argument line by line and tried to break it,
and a checker who compared the statement with the page of the book, looked up every cited
result, checked for circular use of later results, and tested every explicit formula by
special cases and simulation. A solution was revised until both accepted it.

## Building

Compile from this directory with

```bash
pdflatex KaratzasShreveSolutions.tex
```

two or three times, so that the contents page numbers and cross-references settle.
The document needs no local style file.

## Numerical checks (`code/`)

Where an item asks for an explicit distribution, density, transform, moment or
probability, the script `code/cN_S_XX.py` (chapter N, item S.XX) checks the formula
against quadrature or a seeded Monte Carlo simulation. Each script is standalone
(NumPy and SciPy only), runs in well under two minutes, and prints the quantities it
compares. Agreement is up to sampling or discretisation error, which the scripts report.
