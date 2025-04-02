---
layout: post
title: "A Fast Method for Finding the Overlap of N-Rectangles"
categories: "Mathematics"
featImg: safeguard_flowchart.png
excerpt: "Designing an algorithm for finding the area of N intersecting rectangles"
permalink: "n-overlapping-rectangles-algorithm"
style: 
---

# Motivating Problem
Consider a set of $$ N $$ rectangles in 2d-space. The goal is simple: we want to find the total area of all overlapping regions. 

[insert image later]

We can parametrize each rectangle with two coordinate pairs

$$ 
B_i = B_i((x_1, y_1), (x_2, y_2))
$$

where $$ (x_1, y_1) $$ define the upper left corner and $$ (x_2, y_2) $$ define the lower right corner
for some rectangle $$ B_i $$ , where $$ i \leq N $$. 

For the simplest case of $$ N = 2 $$, the task is trivial. For an overlap to occur, we first require

1. $$ 

