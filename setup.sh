#!/bin/bash

value=${1:-.}
echo $value
mkdir -p ${value}/results
mkdir -p ${value}/results/data
mkdir -p ${value}/results/logs
mkdir -p ${value}/results/models
mkdir -p ${value}/results/plots
mkdir -p ${value}/results/evals