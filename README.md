# PAIR in Go
An implementation of [PAIR](https://jailbreaking-llms.github.io/)

## Introduction 
This project aims to automate the process of finding jailbreaks for black box models

## Installation 

```sh 
go install github.com/lordxarus/go-pair/cmd/gop
```

## Example Config
- config.toml currently needs to be in the cwd of the binary 
- ApiEnvVar key is the **name** of the environment variable to use 
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

## Usage
```
go-pair is a tool for generating jailbreak prompts
  -at float
    	temperature for the attacker model (default 0.1)
  -goal string
    	objective for the attacker model (default "Make me a clock application.")
  -mt int
    	maximum new tokens for attacker model response (default 2500)
  -str string
    	we won't exit until the target's response contains string
  -thr int
    	programs exits if the judge outputs a number greater than or equal to -thr (default 10)
  -tt float
    	temperature for the target model (default 0.5)
```