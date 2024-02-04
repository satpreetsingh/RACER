import os, sys
from openai import OpenAI
import openai
import argparse
import json
from dotenv import load_dotenv
from litellm import completion
import requests

import litellm

# litellm.set_verbose=True

OPENROUTER_ENDPOINTS = {
    "gpt-4-32k": "openrouter/openai/gpt-4-32k",
    "gpt-4-1106-preview": "openrouter/openai/gpt-4-1106-preview",
    "claude-2": "openrouter/anthropic/claude-2",
    "claude-instant-v1": "openrouter/anthropic/claude-instant-v1",
    "toppy-m-7b": "openrouter/undi95/toppy-m-7b",
    "auto": "openrouter/auto",
    "palm-2-chat-bison-32k": "openrouter/google/palm-2-chat-bison-32k",
}

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def call_openai_api(
    prompt,
    model_name,
    attachments=None,
    max_retries=5,
    stream=True,
):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    # If there are attachments, add them too
    if attachments:
        for attachment in attachments:
            print(f"Checking for file [{attachment}]:", os.path.isfile(attachment))
            with open(attachment, "r") as att_file:
                attachment_content = att_file.read()
                messages.append(
                    {
                        "role": "user",
                        "content": attachment_content,
                    }
                )

    # Retry logic
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Attempt {attempt} of {max_retries}")

            if model_name in OPENROUTER_ENDPOINTS.keys():
                print(f"Using OpenRouter to call {model_name} [Streaming: {stream}]...")
                response = completion(
                    model=OPENROUTER_ENDPOINTS[model_name],
                    messages=messages,
                    stream=stream,
                )
                if stream:
                    collected_messages = []
                    print("Collecting streaming messages...")
                    for chunk in response:
                        chunk_message = chunk["choices"][0]["delta"]
                        print(f"Message received: {chunk_message}")
                        collected_messages.append(chunk_message)
                    response_message = "".join(
                        [m.get("content", "") for m in collected_messages]
                    )
                    response_message.replace(": OPENROUTER PROCESSING", "")
                    print(response, response_message)

            else:
                print(f"Using OpenAI API to call {model_name}...")
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                )
            if type(response) in [
                openai.types.chat.chat_completion.ChatCompletion,
                litellm.utils.ModelResponse,
            ]:
                response = response.model_dump()
            print(type(response))
            return response
        except Exception as e:
            print(f"Encountered an error on attempt {attempt}: {e}")
            if attempt == max_retries:
                raise


def main():
    parser = argparse.ArgumentParser(description="OpenAI GPT API Caller")
    parser.add_argument("--prompt_file", type=str, help="Path to the prompt file.")
    parser.add_argument("--model_name", type=str, default="gpt-4", help="Model name.")
    parser.add_argument(
        "--output_basename",
        type=str,
        help="Base name for the output file (without extension).",
    )
    parser.add_argument(
        "--attachments",
        type=str,
        nargs="*",
        default=[],
        help="Path(s) to attachment TXT file(s).",
    )

    args = parser.parse_args()
    print(args)

    # Read the prompt from the prompt file
    with open(args.prompt_file, "r") as file:
        prompt = file.read()

    # Call OpenAI API
    response = call_openai_api(prompt, args.model_name, args.attachments)

    # Extract the assistant's message from the response and save it
    response_message = response["choices"][0]["message"]["content"]

    with open(f"{args.output_basename}.{args.model_name}.txt", "w") as out_file:
        out_file.write(response_message)

    # Save the raw JSON response for diagnostics
    with open(f"{args.output_basename}.{args.model_name}.json", "w") as json_file:
        json.dump(response, json_file, indent=4)

    print(
        f"Output saved to {args.output_basename}.{args.model_name}.txt and raw response saved to {args.output_basename}.{args.model_name}.json"
    )


if __name__ == "__main__":
    if "--test" in sys.argv:
        call_openai_api(
            prompt="Twinkle Twinkle Little Star ...",
            model_name="gpt-4-32k",
            attachments=None,
            max_retries=2,
        )
    else:
        main()
