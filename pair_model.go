package main

import (
	"context"
	"fmt"
	"time"

	"github.com/tmc/langchaingo/llms"
)

type attTmplVals struct {
	Objective string
	Start     string
}

type judTmplVals struct {
	User   string
	Target string
}

type pairModel struct {
	model llms.Model
	hist  []llms.MessageContent
	// addHistFn modifies hist in place
	ctx     context.Context
	llmOpts []llms.CallOption
}

type ModelOption func(*pairModel)

func WithCtx(ctx context.Context) ModelOption {
	return func(pm *pairModel) {
		pm.ctx = ctx
	}
}

func WithHist(hist []llms.MessageContent) ModelOption {
	return func(pm *pairModel) {
		pm.hist = hist
	}
}

func WithCallOpts(opts ...llms.CallOption) ModelOption {
	return func(pm *pairModel) {
		pm.llmOpts = append(pm.llmOpts, opts...)
	}
}

func (pm *pairModel) Query(q string) (string, error) {
	pm.hist = append(pm.hist, llms.TextParts(llms.ChatMessageTypeHuman, q))
	completion, err := pm.model.GenerateContent(pm.ctx, pm.hist, pm.llmOpts...)
	if err != nil {
		return "", fmt.Errorf("failed during query: %w", err)
	}
	if len(completion.Choices) == 0 {
		return "", fmt.Errorf("no completion choices")
	}
	pm.hist = append(pm.hist, llms.TextParts(llms.ChatMessageTypeAI, completion.Choices[0].Content))

	return completion.Choices[0].Content, nil
}

func NewPairModel(model llms.Model, opts ...ModelOption) *pairModel {
	pm := &pairModel{
		model:   model,
		hist:    []llms.MessageContent{},
		ctx:     context.Background(),
		llmOpts: []llms.CallOption{},
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
