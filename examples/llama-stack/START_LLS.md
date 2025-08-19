# Options for starting Llama Stack

Depending on the choice of the inference server, the vector database, and other configurations, there are multiple ways to start the Llama Stack backend server. This documentation page explains a few alternatives.

As a simple starting point, we will use the [starter distribution](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/starter.html) which allows Llama Stack to easily run locally.

## Using LM Studio for inference

Since [LM Studio](https://lmstudio.ai/) is exposing an openai-compatible api, we can use it as a local inference server.
Since Llama Stack does not have a native provider for it, we connect it as a vllm server.

1. Install and run [LM Studio](https://lmstudio.ai/).

2. Run Llama Stack.

    ```shell
    export LLAMA_STACK_PORT=8321

    podman run \
        -it \
        --pull always \
        -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
        -v ~/.llama:/root/.llama \
        llamastack/distribution-starter:0.2.16 \
        --port $LLAMA_STACK_PORT \
        --env ENABLE_MILVUS=milvus \
        --env ENABLE_OLLAMA=ollama \
        --env OLLAMA_INFERENCE_MODEL=$INFERENCE_MODEL \
        --env OLLAMA_URL=http://host.containers.internal:11434 \
        --env ENABLE_vllm=lms \
        --env VLLM_URL=http://host.containers.internal:1234/v1
    ```

    _Note: Ollama is also enabled, because the current release does not yet to run without. It will be fixed soon._

3. Register models (should have been downloaded in LM Studio already)

    ```sh
    uvx --from llama-stack-client llama-stack-client models register lms/llama-3.2-3b-instruct --provider-id vllm --provider-model-id llama-3.2-3b-instruct

    uvx --from llama-stack-client llama-stack-client models register lms/openai/gpt-oss-20b --provider-id vllm --provider-model-id openai/gpt-oss-20b

    uvx --from llama-stack-client llama-stack-client models register lms/ibm/granite-3.2-8b --provider-id vllm --provider-model-id ibm/granite-3.2-8b
    ```

4. Test the models.

    ```sh
    uvx --from llama-stack-client llama-stack-client --endpoint http://localhost:8321 \
        inference chat-completion \
        --model-id vllm/llama-3.2-3b-instruct \
        --message "Write a short story about a robot."
    ```

## Using watsonx.ai for inference

For connecting the Llama Stack starter distribution to the watsonx.ai inference service, we will rely on running [LiteLLM](https://www.litellm.ai/) as a proxy, such that we can serve the inference endpoint as an openai-compatible api.


1. Create file `env.secrets` with your watsonx.ai credentials.

    ```sh
    WATSONX_URL=https://us-south.ml.cloud.ibm.com
    WATSONX_APIKEY=apikey
    WATSONX_PROJECT_ID=proj
    ```

2. Create a `litellm_config.yaml` file with the models to proxy

    ```yaml
    model_list:
      - model_name: "granite-3-3-8b-instruct"
        litellm_params:
          model: "watsonx/ibm-granite/granite-3-3-8b-instruct"
      - model_name: "granite-vision-3.2-2b"
        litellm_params:
          model: "watsonx/ibm-granite/granite-vision-3.2-2b"
      - model_name: "gpt-oss-120b"
        litellm_params:
          model: "watsonx/openai/gpt-oss-120b"
      - model_name: "llama-3-3-70b"
        litellm_params:
          model: "watsonx/meta-llama/llama-3-3-70b-instruct"
      - model_name: "Llama-4-Maverick" 
        litellm_params:
          model: "watsonx/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
      - model_name: "*" 
        litellm_params:
          model: "*"
    ```

3. Run LiteLLM.

    ```sh
    podman run \
      -v $(pwd)/litellm_config.yaml:/app/config.yaml \
      --rm \
      -p 4000:4000 \
      --env-file env.secrets \
      ghcr.io/berriai/litellm:v1.74.9-stable \
      --config /app/config.yaml
    ```

2. Run Llama Stack.

    ```shell
    export LLAMA_STACK_PORT=8321

    podman run \
        -it \
        --pull always \
        -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
        -v ~/.llama:/root/.llama \
        llamastack/distribution-starter:0.2.16 \
        --port $LLAMA_STACK_PORT \
        --env ENABLE_MILVUS=milvus \
        --env ENABLE_OLLAMA=ollama \
        --env OLLAMA_INFERENCE_MODEL=$INFERENCE_MODEL \
        --env OLLAMA_URL=http://host.containers.internal:11434 \
        --env ENABLE_vllm=wx \
        --env VLLM_URL=http://host.containers.internal:4000
    ```

    _Note: Ollama is also enabled, because the current release does not yet to run without. It will be fixed soon._

3. Register models (should have been downloaded in LM Studio already)

    ```sh
    uvx --from llama-stack-client llama-stack-client models register wx/llama-3-3-70b --provider-id vllm --provider-model-id llama-3-3-70b

    uvx --from llama-stack-client llama-stack-client models register wx/Llama-4-Maverick --provider-id vllm --provider-model-id Llama-4-Maverick

    uvx --from llama-stack-client llama-stack-client models register wx/gpt-oss-120b --provider-id vllm --provider-model-id gpt-oss-120b
    ```

4. Test the models.

    ```sh
    uvx --from llama-stack-client llama-stack-client --endpoint http://localhost:8321 \
        inference chat-completion \
        --model-id wx/gpt-oss-120b \
        --message "Write a short story about a robot."
    ```

