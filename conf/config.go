package conf

import (
	"fmt"
	"os"

	"github.com/BurntSushi/toml"
)

type ModelType int

//go:generate stringer -type=ModelType
const (
	AttackerType ModelType = iota
	JudgeType
	TargetType
)

type Config struct {
	Type ModelType `toml:"type"`
	// This is the env variable that will be loaded not the key itself.
	APIEnvVar string `toml:"api_key"`
	BaseURL   string `toml:"base_url"`
	ModelName string `toml:"model_version"`
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
