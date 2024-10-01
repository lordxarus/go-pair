package pair

import (
	"fmt"
	"log"
	"os"

	"github.com/BurntSushi/toml"
	"github.com/tmc/langchaingo/llms/huggingface"
	"github.com/tmc/langchaingo/llms/openai"
)

type ModelType int

//go:generate stringer -type=ModelType
const (
	AttackerType ModelType = iota
	JudgeType
	TargetType
)

type Model struct {
	Type ModelType
	// This is the env variable that will be loaded not the key itself.
	ApiEnvVar string
	BaseUrl   string
	ModelName string
}

func (conf *Model) ToOaiLlm() *openai.LLM {
	m, err := openai.New(
		openai.WithModel(conf.ModelName),
		openai.WithBaseURL(conf.BaseUrl),
		openai.WithToken(os.Getenv(conf.ApiEnvVar)),
	)
	if err != nil {
		log.Fatalf("failed to create %s model: %s", conf.Type.String(), err.Error())
	}
	return m
}

func (conf *Model) ToHfLlm() *huggingface.LLM {
	m, err := huggingface.New(
		huggingface.WithModel(conf.ModelName),
		huggingface.WithToken(os.Getenv(conf.ApiEnvVar)),
	)
	if err != nil {
		log.Fatalf("failed to create %s model: %s", conf.Type.String(), err.Error())
	}
	return m
}

type Config struct {
	Models map[string]*Model
}

func LoadConfig(path string) (*Config, error) {
	var cfg Config
	if _, err := toml.DecodeFile(path, &cfg); err != nil {
		return nil, fmt.Errorf("failed to load config: %w", err)
	}
	return &cfg, nil
}

func SaveConfig(path string, cfg *Config) error {
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create config file: %w", err)
	}
	defer file.Close()

	encoder := toml.NewEncoder(file)
	if err := encoder.Encode(cfg); err != nil {
		return fmt.Errorf("failed to encode config: %w", err)
	}
	return nil
}
