#!/bin/bash

for i in {1..1000}; do
    output=$(cat /sys/bus/i2c/drivers/ina3221x/7-0040/iio\:device0/in_power1_input)  # Replace 'your_program' with the actual command to run your program
    sleep 0.1
    echo "$output"
done