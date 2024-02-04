#!/bin/bash
output_file="../metadata/transcript_lengths.csv"
echo "filename,subject_id,num_words,num_lines,num_characters" > "$output_file"

# Loop through each .txt file
for file in ../data/otter_export_100/*.txt; do
    # echo $file
    filename=$(basename "$file")
    subject_id=${filename:0:4}
    num_words=$(wc -w < "$file" | tr -d '[:space:]')
    num_lines=$(wc -l < "$file" | tr -d '[:space:]')
    num_characters=$(wc -m < "$file" | tr -d '[:space:]')
    echo "$filename,$subject_id,$num_words,$num_lines,$num_characters" >> "$output_file"
done

echo "Done! Head of $output_file"
head "$output_file"
