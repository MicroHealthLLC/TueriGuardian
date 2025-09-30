# API Client

## Python

## TEST

=== "Synchronous"

    ```python linenums="1"
    import os
    import requests

    TUERI_API_KEY = os.environ.get("TUERI_API_KEY")
    TUERI_BASE_URL = os.environ.get("TUERI_URL")

    class TueriMaliciousPromptException(Exception):
        scores = {}

        def __init__(self, *args, **kwargs):
            super().__init__(*args)
            self.scores = kwargs.get("scores", {})

        def __str__(self):
            scanners = [scanner for scanner, score in self.scores.items() if score > 0]

            return f"Tueri detected a malicious prompt. Scanners triggered: {', '.join(scanners)}; scores: {self.scores}"


    class TueriRequestException(Exception):
        pass

    def request_tueri_prompt(prompt: str):
        try:
            response = requests.post(
                url=f"{TUERI_BASE_URL}/analyze/prompt",
                json={"prompt": prompt},
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {TUERI_API_KEY}",
                },
            )

            response_json = response.json()
        except requests.RequestException as err:
            raise TueriRequestException(err)

        if not response_json["is_valid"]:
            raise TueriMaliciousPromptException(scores=response_json["scanners"])

        return response_json["sanitized_prompt"]

    prompt = "Write a Python function to calculate the factorial of a number."
    sanitized_prompt = request_TUERI_prompt(prompt)
    print(sanitized_prompt)
    ```

=== "Call LLM provider and Tueri API in parallel"

    ```python linenums="1"
    import os
    import asyncio
    import aiohttp
    from openai import AsyncOpenAI

    TUERI_API_KEY = os.environ.get("TUERI_API_KEY")
    TUERI_BASE_URL = os.environ.get("TUERI_URL")
    openai_client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    system_prompt = "You are a Python tutor."

    class TueriMaliciousPromptException(Exception):
        scores = {}

        def __init__(self, *args, **kwargs):
            super().__init__(*args)
            self.scores = kwargs.get("scores", {})

        def __str__(self):
            scanners = [scanner for scanner, score in self.scores.items() if score > 0]

            return f"LLM Guard detected a malicious prompt. Scanners triggered: {', '.join(scanners)}; scores: {self.scores}"


    class TueriRequestException(Exception):
        pass

    async def request_openai(prompt: str) -> str:
        chat_completion = await openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": prompt},
            ],
            model="gpt-3.5-turbo",
        )

        return chat_completion.choices[0].message.content


    async def request_tueri_prompt(prompt: str):
        async with aiohttp.ClientSession() as session:
            try:
                response = await session.post(
                    url=f"{TUERI_BASE_URL}/analyze/prompt",
                    json={"prompt": prompt},
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {TUERI_API_KEY}",
                    },
                    ssl=False,
                    raise_for_status=True,
                )

                response_json = await response.json()
            except Exception as e:
                raise TueriRequestException(e)

            if not response_json["is_valid"]:
                raise TueriMaliciousPromptException(scores=response_json["scanners"])

    async def generate_completion(prompt: str) -> str:
        result = await asyncio.gather(
            request_tueri_prompt(prompt),
            request_openai(prompt),
        )

        return result[1]

    prompt = "Write a Python function to calculate the factorial of a number."
    message = asyncio.run(
        generate_completion(prompt)
    )
    ```
