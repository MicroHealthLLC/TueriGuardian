# Apply Tueri Content Mod across 100+ LLMs w/ LiteLLM

Use Tueri with LiteLLM Proxy to moderate calls across Anthropic/Bedrock/Gemini/etc. LLMs with [LiteLLM](https://github.com/BerriAI/litellm)


LiteLLM currently supports requests in:
- [The OpenAI format](https://docs.litellm.ai/docs/completion/input) - `/chat/completion`, `/embedding`, `completion`, `/audio/transcription`, etc.
- [The Anthropic format](https://docs.litellm.ai/docs/anthropic_completion) - `/messages`


[**Detailed Docs**](https://docs.litellm.ai/docs/proxy/quick_start)

## Pre-Requisites
- Install litellm proxy - `pip install 'litellm[proxy]'`
- Setup [Tueri Docker](../api/deployment.md#from-docker)

## Quick Start

Let's add Tueri content mod for Anthropic API calls

Set the Tueri API Base in your environment

```bash
export TUERI_API_BASE="http://0.0.0.0:8192" # deployed Tueri api
export ANTHROPIC_API_KEY="sk-..." # anthropic api key
```

Add `tueri_moderations` as a callback in a config.yaml

```yaml
model_list:
  - model_name: claude-3.5-sonnet ### RECEIVED MODEL NAME ###
    litellm_params: # all params accepted by litellm.completion() - https://docs.litellm.ai/docs/completion/input
      model: anthropic/claude-3-5-sonnet-20240620 ### MODEL NAME sent to `litellm.completion()` ###
      api_key: os.environ/ANTHROPIC_API_KEY


litellm_settings:
    callbacks: ["tueri_moderations"]
```

Now you can easily test it:

```bash
litellm --config /path/to/config.yaml
```

- Make a regular /chat/completion call

- Check your proxy logs for any statement with `Tueri:`

Expected results:

```bash
Tueri: Received response - {"sanitized_prompt": "hello world", "is_valid": true, "scanners": { "Regex": 0.0 }}
```
### Turn on/off per key


**1. Update config**

```yaml
model_list:
  - model_name: claude-3.5-sonnet ### RECEIVED MODEL NAME ###
    litellm_params: # all params accepted by litellm.completion() - https://docs.litellm.ai/docs/completion/input
      model: anthropic/claude-3-5-sonnet-20240620 ### MODEL NAME sent to `litellm.completion()` ###
      api_key: os.environ/ANTHROPIC_API_KEY

litellm_settings:
    callbacks: ["tueri_moderations"]
    tueri_mode: "key-specific"

general_settings:
    database_url: "postgres://.." # postgres db url
    master_key: "sk-1234"
```

**2. Create new key**

```bash
curl --location 'http://localhost:4000/key/generate' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data '{
    "models": ["claude-3.5-sonnet"],
    "permissions": {
        "enable_tueri_check": true # 👈 KEY CHANGE
    }
}'

# Returns {..'key': 'my-new-key'}
```

**3. Test it!**

```bash
curl --location 'http://0.0.0.0:4000/v1/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer my-new-key' \ # 👈 TEST KEY
--data '{"model": "claude-3.5-sonnet", "messages": [
        {"role": "system", "content": "Be helpful"},
        {"role": "user", "content": "What do you know?"}
    ]
    }'
```

### Turn on/off per request

**1. Update config**
```yaml
litellm_settings:
    callbacks: ["tueri_moderations"]
    tueri_mode: "request-specific"
```

**2. Create new key**

```bash
curl --location 'http://localhost:4000/key/generate' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data '{
    "models": ["claude-3.5-sonnet"],
}'

# Returns {..'key': 'my-new-key'}
```

**3. Test it!**

### OpenAI SDK

```python
import openai
client = openai.OpenAI(
    api_key="sk-1234",
    base_url="http://0.0.0.0:4000"
)

# request sent to model set on litellm proxy, `litellm --model`
response = client.chat.completions.create(
    model="claude-3.5-sonnet",
    messages = [
        {
            "role": "user",
            "content": "this is a test request, write a short poem"
        }
    ],
    extra_body={ # pass in any provider-specific param, if not supported by openai, https://docs.litellm.ai/docs/completion/input#provider-specific-params
        "metadata": {
            "permissions": {
                "enable_tueri_check": True # 👈 KEY CHANGE
            },
        }
    }
)

print(response)
```

### Curl

```bash
curl --location 'http://0.0.0.0:4000/v1/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer my-new-key' \ # 👈 TEST KEY
--data '{"model": "claude-3.5-sonnet", "messages": [
        {"role": "system", "content": "Be helpful"},
        {"role": "user", "content": "What do you know?"}
    ]
    }'
```
