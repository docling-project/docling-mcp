# Llama Stack Playground

In this directory we create a container image of the [Llama Stack Playground](https://github.com/meta-llama/llama-stack/tree/main/llama_stack/distribution/ui).

## Build the image

```shell
IMAGE_NAME=llama-stack-playground
podman build -t ${IMAGE_NAME}  .
```

## Run the image

```shell
podman run -e LLAMA_STACK_ENDPOINT="http://host.containers.internal:8321" -p 8501:8501 ${IMAGE_NAME}
```
