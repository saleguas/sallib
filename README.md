# saltools
A library of helpful python functions I find myself using quite often

```
.\.venv\Scripts\Activate.ps1

pip install -e .[dev]

cd /mnt/c/Users/salva/Documents/sallib
```

```
This is a library i am making for a plethora of useful functions and tools that I may use. It is a python library so it is important that you maintain the correct structure.

I need you to create a new module for AI querying. This is for sending AI queries to OpenAI, Anthropic, Gemini, grok, etc. It should be able to support different models with a similar interface. Perhaps we can break it up into multiple private functions such as _query_openai, _query_anthropic, _query_gemini, etc. and then have one main function that has a parameters such as model, model_type (or whatever the exact model sic alled). prompt, image(optional), temperature, api_key (optional)

It should optionally be able to support image generation, as well, which should be psased in as a path.

It should also be able to support passing in optioally an API key, it should by default be none and look for the API key as OPENAI_API_KEY, ANTHROPIC_API_KEY. Also have a function to debug useful information, such as how mnay tokens we are sending, how much this query cost, etc. in some fun colors as well

