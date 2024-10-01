package pair

import (
	"context"
	"fmt"
	"time"

	"github.com/tmc/langchaingo/llms"
)

type AttTmplVals struct {
	Objective string
	Start     string
}

type JudTmplVals struct {
	User   string
	Target string
}

type PairModel struct {
	Model llms.Model
	Hist  []llms.MessageContent
	// addHistFn modifies hist in place
	Ctx     context.Context
	LlmOpts []llms.CallOption
}

type ModelOption func(*PairModel)

func WithCtx(ctx context.Context) ModelOption {
	return func(pm *PairModel) {
		pm.Ctx = ctx
	}
}

func WithHist(hist []llms.MessageContent) ModelOption {
	return func(pm *PairModel) {
		pm.Hist = hist
	}
}

func WithCallOpts(opts ...llms.CallOption) ModelOption {
	return func(pm *PairModel) {
		pm.LlmOpts = append(pm.LlmOpts, opts...)
	}
}

func (pm *PairModel) Query(q string) (string, error) {
	pm.Hist = append(pm.Hist, llms.TextParts(llms.ChatMessageTypeHuman, q))
	completion, err := pm.Model.GenerateContent(pm.Ctx, pm.Hist, pm.LlmOpts...)
	if err != nil {
		return "", fmt.Errorf("failed during query: %w", err)
	}
	if len(completion.Choices) == 0 {
		return "", fmt.Errorf("no completion choices")
	}
	pm.Hist = append(pm.Hist, llms.TextParts(llms.ChatMessageTypeAI, completion.Choices[0].Content))

	return completion.Choices[0].Content, nil
}

func NewPairModel(model llms.Model, opts ...ModelOption) *PairModel {
	pm := &PairModel{
		Model:   model,
		Hist:    []llms.MessageContent{},
		Ctx:     context.Background(),
		LlmOpts: []llms.CallOption{},
	}

	for _, opt := range opts {
		opt(pm)
	}

	return pm
}

type TokenTracker struct {
	tokens    int
	startTime time.Time
	ticker    *time.Ticker
}

func NewTokenTracker() *TokenTracker {
	tt := &TokenTracker{
		startTime: time.Now(),
		ticker:    time.NewTicker(time.Minute),
	}

	go func() {
		for range tt.ticker.C {
			tt.Reset()
		}
	}()

	return tt
}

func (tt *TokenTracker) AddTokens(count int) {
	tt.tokens += count
}

func (tt *TokenTracker) TokensPerMinute() float64 {
	elapsed := time.Since(tt.startTime).Minutes()
	if elapsed == 0 {
		return 0
	}
	return float64(tt.tokens) / elapsed
}

func (tt *TokenTracker) Reset() {
	tt.tokens = 0
	tt.startTime = time.Now()
}
