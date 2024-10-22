#!/bin/bash

# Initialize variables to store the sum of each value
sum1=0
sum2=0
sum3=0

# Run the program 100 times
for i in {1..100}; do
    output=$(./softmax_cuda_float_256 -s 1000 -c)  # Replace 'your_program' with the actual command to run your program
    IFS=';' read -ra values <<< "$output"
    sum1=$(awk "BEGIN {print $sum1 + ${values[0]}}")
    sum2=$(awk "BEGIN {print $sum2 + ${values[1]}}")
    sum3=$(awk "BEGIN {print $sum3 + ${values[2]}}")
done

# Calculate the average for each value
average1=$(awk "BEGIN {print $sum1 / 100}")
average2=$(awk "BEGIN {print $sum2 / 100}")
average3=$(awk "BEGIN {print $sum3 / 100}")

# Print the averages
echo "Elapsed time Host->Device: $average1"
echo "Elapsed time kernel: $average2"
echo "Elapsed time Device->Host: $average3"
