# MODEL_NAME=gpt-3.5-turbo-16k
MODEL_NAME=gpt-4-32k
DATA_DIR=../data/otter_export_100/
PROMPT_FILE=prompts/prompt_8.txt
SORTED_FILES=$(find $DATA_DIR -name "*.txt" | sort)
echo "$SORTED_FILES" > /tmp/sorted_files.txt


# for TRANSCRIPT in $SORTED_FILES; do
#     FILE_CODE=$(echo ${TRANSCRIPT} | grep -o 'C[0-9]\{3\}')
#     echo ${FILE_CODE} ${TRANSCRIPT}

#     python llm.py --prompt_file $PROMPT_FILE \
#         --attachments $TRANSCRIPT \
#         --model_name $MODEL_NAME \
#         --output_basename outputs/${FILE_CODE}
# done

# Using GNU Parallel
parallel '
    FILE_CODE=$(echo {} | grep -o "C[0-9]\{3\}")
    echo ${FILE_CODE} {}
    python llm.py --prompt_file '"$PROMPT_FILE"' \
        --attachments {} \
        --model_name '"$MODEL_NAME"' \
        --output_basename outputs/${FILE_CODE}
' ::: $SORTED_FILES