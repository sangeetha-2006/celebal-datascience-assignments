# Week 8 Assignment – Single Agent Pipeline

## Overview

This project implements a simple **Single-Agent Smart Assistant** in Python. The agent identifies the user's intent and routes the query to the appropriate tool.

## Features

* Calculator tool for mathematical expressions
* Keyword extraction tool
* General response for other queries
* Conditional routing
* Basic error handling
* Interactive mode

## Tools Used

* Python
* `re` (Regular Expressions)
* `json`

## Routing Logic

* If the query contains **"calculate"** → Uses the Calculator tool.
* If the query contains **"keywords"** → Uses the Keyword Extractor.
* Otherwise → Returns a general response.

## Sample Queries

* `Calculate 20 + 5`
* `Extract keywords from Artificial Intelligence is transforming industries`
* `What is machine learning?`

## Output

The agent returns a structured JSON-like response containing:

* `type`
* `result`

## Conclusion

This assignment demonstrates a basic single-agent pipeline using conditional routing, tool integration, and structured responses in Python.

