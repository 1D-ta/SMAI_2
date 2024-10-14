#!/bin/bash

# Check if an input file is provided
if [ $# -eq 1 ]; then
    # If a file is provided, use input redirection
    python3 tsp_solver_concorde.py < "$1"
else
    # If no file is provided, run normally (will read from stdin)
    python3 tsp_solver_concorde.py
fi