#!/bin/bash

MODEL="models/solar_unet.pth"

for year in 2017 2018 2019 2020 2021 2022 2023 2024; do
    echo "Removing existing representations.."
    rm -rf global_0.1_degree_representation
    echo "Processing year $year..."
    uv run predict_uk.py --model "$MODEL" --output-dir "uk/$year" --year "$year"
done
