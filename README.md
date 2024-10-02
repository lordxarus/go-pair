# PAIR in Go
An implementation of [PAIR](https://jailbreaking-llms.github.io/)

## Introduction 
This project aims to automate the process of finding jailbreaks for black box models

## Installation 

```sh 
go install github.com/lordxarus/go-pair/cmd/gop
```

- Place config in the same folder as the binary, named config.toml
## Example Config
```toml 
[Models.Attacker]
Type = 0
ApiEnvVar = "HF_KEY"
BaseUrl = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-72B-Instruct/v1"
ModelName = "Qwen/Qwen2.5-72B-Instruct"

[Models.Judge]
Type = 1
ApiEnvVar = "OPENROUTER_KEY"
BaseUrl = "https://openrouter.ai/api/v1"
ModelName = "meta-llama/llama-3.2-90b-vision-instruct"

[Models.Target]
Type = 2
ApiEnvVar = "OPENROUTER_KEY"
BaseUrl = "https://openrouter.ai/api/v1"
ModelName = "llama-3.1-sonar-huge-128k-online"
```