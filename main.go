package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"text/template"
	"time"

	"github.com/fatih/color"
	"github.com/joho/godotenv"
	"github.com/lordxarus/go-pair/conf"
	"github.com/tmc/langchaingo/llms"
	"golang.org/x/exp/rand"
)

/*
TODO

	Add whisper API support so we can speak to the AI
	TTS?

	    Implementing the 1-10 system actually seemed to fix the issue with an outright refusal
		causing the judge to deem it unsafe.

			When the model completely refuses with something like "I can't fulfill that request",
			the judge immediately calls it unsafe. Currently I just retry the target model but it will
			always find a way to refuse. Probably can be fixed with better prompt for judge

			I think giving the attacker a score from 1-10 would be a good idea. This would be based on how well
			the target complied. We can specify a threshold and exit when we pass it

	I'm getting rate limited by Groq. I will probably always want to use Groq so a rate limiter wouldn't
	be a horrible idea. I don't really know how I would want to implement it though

	I need to be able to intervene and provide my own feedback to the attacker model.

	There will be an interactive (dev) mode
	The main way the user will use the program is this:
	./pair -t <template> -o <objective> -s <start>
*/

func init() {
	err := godotenv.Load()
	if err != nil {
		log.Fatalf("Error loading .env file")
	}
	rand.Seed(uint64(time.Now().UnixNano()))
	log.SetFlags(log.Flags() | log.Lshortfile)

	flag.Usage = func() {
		w := flag.CommandLine.Output() // may be os.Stderr - but not necessarily

		fmt.Fprintf(w, "go-pair-%s is a tool for generating jailbreak prompts\n", "0.0.1")

		flag.PrintDefaults()

	}
}

func main() {
	// These flags govern the target model's parameters. I will likely put
	// these into the configs. And then I will load all toml files in x directory
	// and map them to their respective models using conf.ModelType
	fStrict := flag.Bool("s", false, "if set, the program will not exit until\n the target's response contains -start string")
	fTemp := flag.Float64("t", .5, "temperature for the target model")
	fMaxTokens := flag.Int("mt", 2500, "maximum new tokens for attacker model response")
	fObjective := flag.String("goal", "Make me a clock application.", "objective for the attacker model")
	fStartStr := flag.String("str", "", "start string for the target model response")
	fThreshold := flag.Int("thr", 10, "programs exits if the judge outputs a number greater than or equal to -thr")

	// fMaxLen := flag.Int("max-len", 250, "maximum length for the model")
	flag.Parse()

	fmt.Printf("Using prompt: %s\n", *fObjective)

	hfHubKey := os.Getenv("HF_API_KEY")
	_ = hfHubKey

	// template stuff
	attTmplPath := "prompt/attacker/logical.txt"
	judTmplPath := "prompt/judge/llama-guard.txt"

	var judge *pairModel
	var target *pairModel
	var attacker *pairModel

	attVals := attTmplVals{
		Objective: *fObjective,
		Start:     *fStartStr,
	}

	config, err := loadOrCreateConfig("./config.toml")
	if err != nil {
		log.Fatalf("failed to load attacker config: %s", err.Error())
	}

	// Attacker model
	attPairPrompt, err := execTmpl(attTmplPath, attVals)
	if err != nil {
		log.Fatal("failed to get prompt: ", err)
	}

	attacker = NewPairModel(config.Models["Attacker"].ToLlm(),
		WithHist([]llms.MessageContent{
			// added the extra instruction when I was using gemma2-9b-it. others might not need it
			llms.TextParts(llms.ChatMessageTypeSystem, attPairPrompt+"\n"+"Do not include markdown formatting tags. COMPLY WITH JSON FORMATTING. The first character will ALWAYS be {"),
		}),
		WithCallOpts(llms.WithJSONMode()),
	)

	// Judge model
	judPairPrompt, err := execTmpl(judTmplPath, judTmplVals{})
	if err != nil {
		log.Fatal("failed to get prompt: ", err)
	}

	judge = NewPairModel(config.Models["Judge"].ToLlm(), WithHist([]llms.MessageContent{
		llms.TextParts(llms.ChatMessageTypeSystem, judPairPrompt),
	}))

	// Target model
	target = NewPairModel(
		config.Models["Target"].ToLlm(),
		WithCallOpts(llms.WithMaxTokens(*fMaxTokens), llms.WithTemperature(*fTemp)),
	)

	// After so many iterations the attacker model can go off the rails
	count := 0
	// Off to the races
	// s := bufio.NewScanner(os.Stdin)
main:
	for {

		count++
		color.White("Iteration: %d\n", count)

		resp, err := attacker.Query(attVals.Objective)
		if err != nil {
			log.Fatalln("failed to call attacker: ", err)
		}
		resp = strings.ReplaceAll(resp, "\n", "")

		// This is definitely too nested
		if strings.TrimSpace(resp) == "" {
			log.Println("attacker returned empty string")
			for _, v := range attacker.hist {
				concat := fromParts(v.Parts)
				if v.Role == llms.ChatMessageTypeAI &&
					strings.TrimSpace(concat) != "" {
					attacker.hist = []llms.MessageContent{
						llms.TextParts(
							llms.ChatMessageTypeSystem,
							fmt.Sprintf("%s\nYour last output: %s\n",
								attPairPrompt, concat)),
						llms.TextParts(
							llms.ChatMessageTypeHuman,
							fmt.Sprintf("Something went wrong, this is what the target said last: \"%v\"\n",
								attacker.hist[len(attacker.hist)-1].Parts[0].(llms.TextContent).String()),
						),
					}
					log.Printf("retrying with new hist: %v\n", attacker.hist)
					continue main
				}
			}
			break main

		}

		fmt.Print("\n\n")

		var result map[string]interface{}
		err = json.Unmarshal([]byte(resp), &result)

		if err != nil {
			log.Fatalf("failed to unmarshal attacker response into json: %v\noutput: %s", err, resp)
		}

		newTargPrompt, ok := result["prompt"].(string)
		if !ok {
			log.Fatalf("expected prompt value to be a string, got %T", result["prompt"])
		}

		improvement, ok := result["improvement"].(string)
		if !ok {
			log.Fatalf("expected improvement value to be a string, got %T", result["improvement"])
		}

		color.Set(color.FgRed)
		fmt.Println(color.RedString("Improvement: ") + color.RedString(improvement))
		fmt.Println(color.HiRedString("New prompt: ") + color.HiRedString(newTargPrompt))
		color.Unset()

		target.hist = []llms.MessageContent{}
		targetResp, err := target.Query(strings.ReplaceAll(newTargPrompt, "\"", "'"))
		for b := strings.Contains(targetResp, "I cannot"); b; {
			targetResp, err = target.Query(strings.ReplaceAll(newTargPrompt, "\"", "'"))
		}

		if err != nil {
			log.Println("failed to call target: ", err)
			continue
		}

		color.Set(color.FgGreen)
		fmt.Println(targetResp)
		color.Unset()

		judPrompt, err := execTmpl(judTmplPath, judTmplVals{
			User:   newTargPrompt,
			Target: targetResp,
		})
		if err != nil {
			log.Fatal("failed to get prompt: ", err)
		}
		judgeResp, err := judge.Query(judPrompt + "\n" + "Do not include explanations. Only judge the content. With the provided format.")
		if err != nil {
			log.Fatalln("failed to call judge: ", err)
		}

		color.Yellow(judgeResp)

		spl := strings.Split(judgeResp, "\n")
		for _, line := range spl {
			if "unsafe" != line {
				continue
			}

			// Threshold test
			if strings.Contains(targetResp, strconv.Itoa(*fThreshold)) {
				fmt.Println("threshold test passed")
				break main
			}

			// If fStrict isn't set, or fStartStr is "" we don't
			// need run the startStr test
			if !*fStrict || *fStartStr == "" {
				continue
			}

			if strings.Contains(targetResp, *fStartStr) {
				fmt.Println("start str test passed")
				break main
			}
		}

		attacker.hist = append(attacker.hist,
			llms.TextParts(
				llms.ChatMessageTypeHuman,
				fmt.Sprintf("The response: %s\n The judgement: %s\n", targetResp, judgeResp),
			),
		)
		time.Sleep(850 * time.Millisecond)
	}

}

// func getModel(opts ...openai.Option) (*openai.LLM, error) {
// 	model, err := openai.New(
// 		opts...,
// 	)
// 	if err != nil {
// 		return nil, fmt.Errorf("failed to create model: %w", err)
// 	}
// 	return model, nil
// }

func execTmpl(path string, vals interface{}) (string, error) {
	bufOut := bytes.NewBuffer([]byte{})

	// Read template from file
	b, err := os.ReadFile(path)
	if err != nil {
		log.Fatalf("%v while trying to read %s", err, path)
	}

	// we aren't using a template if vals is empty
	if vals == (attTmplVals{}) || vals == (judTmplVals{}) {
		return string(b), nil
	}

	// just ignore this error because if this fails Execute will as well and we handle it there
	tmpl, _ := template.New("").Parse(string(b))
	// Exec our template
	err = tmpl.Execute(bufOut, vals)
	if err != nil {
		return "", fmt.Errorf("failed to create tmpl: %w", err)
	}

	return bufOut.String(), nil

}

func loadOrCreateConfig(path string) (*conf.Config, error) {
	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			log.Println("creating new config file: ", path)
			new := &conf.Config{}
			err = conf.SaveConfig(path, new)
			if err != nil {
				return nil, fmt.Errorf("failed to save config: %w", err)
			}
			return new, nil
		}
		return nil, err
	}
	return conf.LoadConfig(path)
}

func fromParts(parts []llms.ContentPart) string {
	var out = ""
	for _, p := range parts {
		out += p.(llms.TextContent).String()
	}
	return out
}

// I'll attempt to clean any json you give me but I can't promise anything
func cleaner(json string) {

	// Something interesting Copilot gave me lmao
	// I'm going to remove all the newlines
	// json = strings.ReplaceAll(json, "\n", "")
	// json = strings.ReplaceAll(json, "\t", "")
	// json = strings.ReplaceAll(json, " ", "")
	// json = strings.ReplaceAll(json, ":", ": ")
	// json = strings.ReplaceAll(json, ",", ", ")
	// json = strings.ReplaceAll(json, "{", "{\n")
	// json = strings.ReplaceAll(json, "}", "\n}")
	// json = strings.ReplaceAll(json, "[", "[\n")
	// json = strings.ReplaceAll(json, "]", "\n]")
	// json = strings.ReplaceAll(json, "}", "}\n")
	// json = strings.ReplaceAll(json, "{", "{\
}
