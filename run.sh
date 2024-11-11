#!/bin/bash

echo Index:
read index

case $index in
  1)
    python3 pacman.py -p MinimaxAgent --frameTime 0 -n 5 -l contestClassic -a depth=3
    ;;
  2)
    python3 pacman.py -p AlphaBetaAgent --frameTime 0 -l trickyClassic -n 5 -a depth=2,evalFn=betterEvaluationFunction
    ;;
  3)
    python3 pacman.py -p ProgressiveDeepeningAgent --frameTime 0 -l contestClassic -n 10
    ;;
  4)
    python3 pacman.py -p AStarMinimaxAgent --frameTime 0 -l contestClassic -n 5
    ;;
  5)
    python3 pacman.py -p AStarAlphaBetaAgent --frameTime 0.2 -l contestClassic -n 5 -a depth=3
    ;;
esac
