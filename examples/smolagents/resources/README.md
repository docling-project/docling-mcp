The prompts are obtained from [smolagents](https://github.com/huggingface/smolagents). The tricky part is that the prompts and the the way the tools are serialized is version specific. So far, we have that,

1. for versions <=v1.20, we started from [smolagents](https://github.com/huggingface/smolagents/tree/v1.20-release/src/smolagents/prompts).
2. for versions >v1.20, we started from [smolagents](https://github.com/huggingface/smolagents/src/smolagents/prompts).
