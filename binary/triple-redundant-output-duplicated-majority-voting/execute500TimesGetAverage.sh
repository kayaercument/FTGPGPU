#!/bin/bash

# Initialize variables to store the sum of each value
sum1=0
sum2=0
sum3=0

# Run the program 500 times
for i in {1..500}; do
    output=$(./convolution_2D_cuda_float_256 -s 256 -k 3 -c)  # Replace 'your_program' with the actual command to run your program
    IFS=';' read -ra values <<< "$output"
    sum1=$(awk "BEGIN {print $sum1 + ${values[0]}}")
    sum2=$(awk "BEGIN {print $sum2 + ${values[1]}}")
    sum3=$(awk "BEGIN {print $sum3 + ${values[2]}}")
done

# Calculate the average for each value
average1=$(awk "BEGIN {print $sum1 / 500}")
average2=$(awk "BEGIN {print $sum2 / 500}")
average3=$(awk "BEGIN {print $sum3 / 500}")

# Print the averages
echo "convolution_2D_bench"
echo "Elapsed time Host->Device: $average1"
echo "Elapsed time kernel: $average2"
echo "Elapsed time Device->Host: $average3"

################################################################################
################################################################################
################################################################################

# Initialize variables to store the sum of each value
sum1=0
sum2=0
sum3=0
average1=0
average2=0
average3=0
# Run the program 500 times
for i in {1..500}; do
    output=$(./max_pooling_cuda_float_256 -s 1024 -l 1 -c)  # Replace 'your_program' with the actual command to run your program
    IFS=';' read -ra values <<< "$output"
    sum1=$(awk "BEGIN {print $sum1 + ${values[0]}}")
    sum2=$(awk "BEGIN {print $sum2 + ${values[1]}}")
    sum3=$(awk "BEGIN {print $sum3 + ${values[2]}}")
done

# Calculate the average for each value
average1=$(awk "BEGIN {print $sum1 / 500}")
average2=$(awk "BEGIN {print $sum2 / 500}")
average3=$(awk "BEGIN {print $sum3 / 500}")

# Print the averages
echo ""
echo "max_pooling_bench"
echo "Elapsed time Host->Device: $average1"
echo "Elapsed time kernel: $average2"
echo "Elapsed time Device->Host: $average3"

################################################################################
################################################################################
################################################################################

# Initialize variables to store the sum of each value
sum1=0
sum2=0
sum3=0
average1=0
average2=0
average3=0
# Run the program 500 times
for i in {1..500}; do
    output=$(./matrix_multiplication_cuda_float_256 -s 256 -c)  # Replace 'your_program' with the actual command to run your program
    IFS=';' read -ra values <<< "$output"
    sum1=$(awk "BEGIN {print $sum1 + ${values[0]}}")
    sum2=$(awk "BEGIN {print $sum2 + ${values[1]}}")
    sum3=$(awk "BEGIN {print $sum3 + ${values[2]}}")
done

# Calculate the average for each value
average1=$(awk "BEGIN {print $sum1 / 500}")
average2=$(awk "BEGIN {print $sum2 / 500}")
average3=$(awk "BEGIN {print $sum3 / 500}")

# Print the averages
echo ""
echo "matrix_multiplication_bench"
echo "Elapsed time Host->Device: $average1"
echo "Elapsed time kernel: $average2"
echo "Elapsed time Device->Host: $average3"

################################################################################
################################################################################
################################################################################

# Initialize variables to store the sum of each value
sum1=0
sum2=0
sum3=0
average1=0
average2=0
average3=0
# Run the program 500 times
for i in {1..500}; do
    output=$(./wavelet_transform_cuda_float_256 -s 1024 -c)  # Replace 'your_program' with the actual command to run your program
    IFS=';' read -ra values <<< "$output"
    sum1=$(awk "BEGIN {print $sum1 + ${values[0]}}")
    sum2=$(awk "BEGIN {print $sum2 + ${values[1]}}")
    sum3=$(awk "BEGIN {print $sum3 + ${values[2]}}")
done

# Calculate the average for each value
average1=$(awk "BEGIN {print $sum1 / 500}")
average2=$(awk "BEGIN {print $sum2 / 500}")
average3=$(awk "BEGIN {print $sum3 / 500}")

# Print the averages
echo ""
echo "wavelet_transform"
echo "Elapsed time Host->Device: $average1"
echo "Elapsed time kernel: $average2"
echo "Elapsed time Device->Host: $average3"


################################################################################
################################################################################
################################################################################

# Initialize variables to store the sum of each value
sum1=0
sum2=0
sum3=0
average1=0
average2=0
average3=0
# Run the program 500 times
for i in {1..500}; do
    output=$(./relu_cuda_float_256 -s 1024 -c)  # Replace 'your_program' with the actual command to run your program
    IFS=';' read -ra values <<< "$output"
    sum1=$(awk "BEGIN {print $sum1 + ${values[0]}}")
    sum2=$(awk "BEGIN {print $sum2 + ${values[1]}}")
    sum3=$(awk "BEGIN {print $sum3 + ${values[2]}}")
done

# Calculate the average for each value
average1=$(awk "BEGIN {print $sum1 / 500}")
average2=$(awk "BEGIN {print $sum2 / 500}")
average3=$(awk "BEGIN {print $sum3 / 500}")

# Print the averages
echo ""
echo "relu_bench"
echo "Elapsed time Host->Device: $average1"
echo "Elapsed time kernel: $average2"
echo "Elapsed time Device->Host: $average3"