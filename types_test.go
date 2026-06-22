package protocolbridge

import (
	"encoding/json"
	"testing"
)

func TestLLMRequestJSONShape(t *testing.T) {
	req := LLMRequest{
		Model: "chaitin/gpt-5.5",
		Prompt: []Message{
			{
				Role:  RoleSystem,
				Parts: []Part{{Type: PartText, Text: &TextPart{Text: "You are helpful."}}},
			},
			{
				Role:  RoleUser,
				Parts: []Part{{Type: PartText, Text: &TextPart{Text: "Hello"}}},
			},
		},
		Tools: []Tool{
			{
				Type:        ToolFunction,
				Name:        "get_weather",
				Description: "Get weather by city.",
				InputSchema: map[string]any{"type": "object"},
			},
		},
		ToolChoice: &ToolChoice{Type: ToolChoiceRequired},
		Stream:     true,
	}

	raw, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}

	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}

	if decoded["model"] != "chaitin/gpt-5.5" {
		t.Fatalf("model = %v", decoded["model"])
	}
	if _, ok := decoded["prompt"].([]any); !ok {
		t.Fatalf("prompt has type %T", decoded["prompt"])
	}
	if decoded["stream"] != true {
		t.Fatalf("stream = %v", decoded["stream"])
	}

	tools := decoded["tools"].([]any)
	tool := tools[0].(map[string]any)
	if _, ok := tool["input_schema"]; !ok {
		t.Fatal("tool input_schema is missing")
	}
}

func TestUsageDistinguishesMissingAndZero(t *testing.T) {
	zero := 0
	usage := Usage{InputTokens: &zero}

	raw, err := json.Marshal(usage)
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}

	if string(raw) != `{"input_tokens":0}` {
		t.Fatalf("usage json = %s", raw)
	}
}
