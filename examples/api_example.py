import requests

ENDPOINT = "http://localhost:5001/api" # Please visit this link and read OpenAI documentation. It has a LOT more than what is shown here.

# This is a very basic example of how to use the KoboldCpp API in python.
# For full documentation, you can launch KoboldCpp and read it at http://localhost:5001/api or view the web docs at https://lite.koboldai.net/koboldcpp_api
# Note: KoboldCpp also provides a fully compatible /v1/completions and /v1/chat/completions API. You can use it as a direct replacement for any OpenAI API usecases.
# Refer to https://platform.openai.com/docs/api-reference/completions and https://platform.openai.com/docs/api-reference/chat

payload = {
    "prompt": "Niko the kobold stalked carefully down the alley, his small scaly figure obscured by a dusky cloak that fluttered lightly in the cold winter breeze.",
    "max_context_length": 4096, # The maximum number of tokens in history that the AI can see. Increase for longer inputs.
    "max_length": 128, # How many token to be generated at maximum. It might stop before this if EOS is allowed.
    "rep_pen": 1.1, # Makes outputs less repetitive by penalizing repetition
    "rep_pen_range": 512, # The range to which to apply repetition penalty
    "rep_pen_slope": 0.7, # This number determains the strength of the repetition penalty over time
    "temperature": 0.8, # How random should the AI outputs be? Lower values make output more predictable.
    "top_k": 100, # Keep the X most probable tokens
    "top_p": 0.9, # Top P sampling / Nucleus Sampling, https://arxiv.org/pdf/1904.09751.pdf
    #"sampler_seed": 1337, # Use specific seed for text generation? This helps with consistency across tests.
}

try:
    response = requests.post(f"{ENDPOINT}/v1/generate", json=payload) # Send prompt to API
    if response.status_code == 200:
        results = response.json()['results'] # Set results as JSON response
        text = results[0]['text'] # inside results, look in first group for section labeled 'text'
        print(text)
except Exception as e:
    print(f"An error occurred: {e}")