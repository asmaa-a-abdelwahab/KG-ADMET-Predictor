#!/bin/bash

# Usage: ./split_cypher.sh input_file output_dir chunk_size
input_file=$1
output_dir=$2
chunk_size=$3

# Ensure the output directory exists
mkdir -p "$output_dir"

chunk_counter=0
line_counter=0
chunk_file="${output_dir}/neo4j_part_${chunk_counter}.cypher"

# Read the file line by line and split into chunks
while IFS= read -r line; do
    if (( line_counter == 0 )); then
        chunk_file="${output_dir}/neo4j_part_${chunk_counter}.cypher"
    fi
    
    echo "$line" >> "$chunk_file"
    ((line_counter++))

    if (( line_counter >= chunk_size )); then
        ((chunk_counter++))
        line_counter=0
    fi
done < "$input_file"

if (( line_counter > 0 )); then
    ((chunk_counter++))
fi

echo "File has been split into $chunk_counter chunks."