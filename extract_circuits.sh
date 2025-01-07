#!/bin/bash

TARGET_DIR=GRCS/inst/rectangular/cz_v2

for file in "$TARGET_DIR"/*.tar.*; do
  # Check if the file exists (in case no .tar.gz files are found)
  if [ -e "$file" ]; then
    echo "Extracting $file..."
    tar xvf "$file" -C "$TARGET_DIR"  # Extract the file in the same directory
    echo "$file extracted successfully."
  else
    echo "No .tar.xz files found in $TARGET_DIR."
    break
  fi
done