import os, sys
from openai import OpenAI
import openai
import argparse
import json
from dotenv import load_dotenv
import requests

OPENROUTER_ENDPOINTS = {
    "gpt-3.5-turbo-16k": "openrouter/openai/gpt-3.5-turbo-16k",
    "gpt-4": "openrouter/openai/gpt-4",
    "gpt-4-32k": "openrouter/openai/gpt-4-32k",
    "gpt-4-1106-preview": "openrouter/openai/gpt-4-1106-preview",
    "claude-2": "openrouter/anthropic/claude-2",
    "claude-instant-v1": "openrouter/anthropic/claude-instant-v1",
    "toppy-m-7b": "openrouter/undi95/toppy-m-7b",
    "auto": "openrouter/auto",
    "palm-2-chat-bison-32k": "openrouter/google/palm-2-chat-bison-32k",
}

load_dotenv()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


def call_openai_api(
    prompt,
    model_name,
    attachments=None,
    max_retries=5,
    stream=False,
    debug=False,
):
    if stream:
        print("Streaming is not supported yet.")
    if model_name not in OPENROUTER_ENDPOINTS.keys():
        raise ValueError(f"Model {model_name} not supported by OpenRouter.")

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

    print(f"Using OpenRouter to call {model_name} [Streaming: {stream}]...")
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Attempt {attempt} of {max_retries}")

            completion = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "http://localhost:3000",
                    "X-Title": "OAI-API",
                },
                model=OPENROUTER_ENDPOINTS[model_name].replace("openrouter/", ""),
                messages=messages,
            )

            print(type(completion))
            if type(completion) == openai.types.chat.chat_completion.ChatCompletion:
                completion = completion.model_dump()
            if debug:
                print(completion)

            return completion
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
    response = call_openai_api(prompt, args.model_name, args.attachments, debug=True)

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
