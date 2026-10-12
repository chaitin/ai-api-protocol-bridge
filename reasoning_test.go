package protocolbridge

import (
	"encoding/json"
	"testing"
)

// decodeChatReasoningLevel is a test helper: the level a chat reasoning_effort
// leaves on the unified request.
func decodeChatReasoningLevel(t *testing.T, effort string) *LLMRequest {
	t.Helper()
	body := `{"model":"gpt-5.4","messages":[{"role":"user","content":"Hello"}]`
	if effort != "" {
		body += `,"reasoning_effort":"` + effort + `"`
	}
	body += `}`

	req, err := NewOpenAIChatAdapter().DecodeRequest([]byte(body))
	if err != nil {
		t.Fatalf("DecodeRequest(%s) error = %v", body, err)
	}
	return req
}

// anthropicThinkingFor bridges a request built from body to Anthropic and returns
// the thinking field, or nil when the field is absent.
func anthropicThinkingFor(t *testing.T, body string, maxTokens *int) (map[string]any, []Warning) {
	t.Helper()

	req, err := NewOpenAIChatAdapter().DecodeRequest([]byte(body))
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	if maxTokens != nil {
		req.MaxOutputTokens = maxTokens
	}
	bridge, ok := NewCrossFamilyBridgeForProtocol(ProtocolOpenAIChat, ProtocolAnthropicMessages)
	if !ok {
		t.Fatal("no chat to anthropic bridge")
	}
	raw, err := bridge.EncodeUpstreamRequest(req, EncodeRequestOptions{Model: "claude-sonnet-4"})
	if err != nil {
		t.Fatalf("EncodeUpstreamRequest() error = %v", err)
	}

	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	thinking, _ := decoded["thinking"].(map[string]any)
	return thinking, req.Warnings
}

// The level a caller asked for has to reach the upstream as a budget, not as a
// bare "reasoning is on". Every level used to land on the 1024 floor, so
// reasoning_effort "high" quietly became the least thinking Anthropic offers.
func TestReasoningEffortDrivesTheAnthropicThinkingBudget(t *testing.T) {
	// max_tokens comfortably above every mapped budget, so nothing is clamped.
	roomy := defaultMaxOutputTokens * 4

	cases := []struct {
		effort string
		want   int
	}{
		{"minimal", 1024},
		{"low", 1024},
		{"medium", 2048},
		{"high", 4096},
		{"xhigh", 8192},
	}
	for _, tc := range cases {
		t.Run(tc.effort, func(t *testing.T) {
			req := decodeChatReasoningLevel(t, tc.effort)
			if req.ReasoningEffort != tc.effort {
				t.Fatalf("ReasoningEffort = %q, want the level to survive decode", req.ReasoningEffort)
			}

			thinking, _ := anthropicThinkingFor(t, `{"model":"gpt-5.4","max_completion_tokens":16384,"reasoning_effort":"`+tc.effort+`","messages":[{"role":"user","content":"Hello"}]}`, &roomy)
			if thinking == nil {
				t.Fatalf("no thinking field for effort %q", tc.effort)
			}
			if thinking["type"] != "enabled" {
				t.Fatalf("thinking = %+v", thinking)
			}
			if got := thinking["budget_tokens"]; got != float64(tc.want) {
				t.Fatalf("budget_tokens = %v, want %d", got, tc.want)
			}
		})
	}
}

func TestReasoningEffortNoneDoesNotEnableThinking(t *testing.T) {
	thinking, _ := anthropicThinkingFor(t, `{"model":"gpt-5.4","reasoning_effort":"none","messages":[{"role":"user","content":"Hello"}]}`, nil)
	if thinking != nil {
		t.Fatalf("thinking = %+v, want none", thinking)
	}

	req := decodeChatReasoningLevel(t, "none")
	if req.Reasoning == nil || *req.Reasoning {
		t.Fatalf("Reasoning = %v, want disabled", req.Reasoning)
	}
}

// Anthropic rejects a budget that does not leave room under max_tokens. A budget
// derived from a level is clamped into the room available, because asking for
// "high" must not turn thinking off; an explicit budget is the caller's own
// number, so the loss is reported instead of rewritten.
func TestThinkingBudgetIsClampedOrReported(t *testing.T) {
	t.Run("derived budget is clamped to fit", func(t *testing.T) {
		limit := 2048
		thinking, warnings := anthropicThinkingFor(t, `{"model":"gpt-5.4","reasoning_effort":"high","messages":[{"role":"user","content":"Hello"}]}`, &limit)
		if thinking == nil {
			t.Fatal("thinking was dropped instead of clamped")
		}
		budget := int(thinking["budget_tokens"].(float64))
		if budget >= limit {
			t.Fatalf("budget = %d, want it under max_tokens %d", budget, limit)
		}
		if budget < minAnthropicThinkingBudgetTokens {
			t.Fatalf("budget = %d, below the Anthropic floor", budget)
		}
		if len(warnings) != 0 {
			t.Fatalf("warnings = %+v, want none for a clamped budget", warnings)
		}
	})

	t.Run("explicit budget that does not fit is reported", func(t *testing.T) {
		req := decodeChatReasoningLevel(t, "")
		limit := 2048
		budget := 8192
		req.MaxOutputTokens = &limit
		req.Reasoning = boolPtr(true)
		req.ReasoningBudgetTokens = &budget

		bridge, _ := NewCrossFamilyBridgeForProtocol(ProtocolOpenAIChat, ProtocolAnthropicMessages)
		raw, err := bridge.EncodeUpstreamRequest(req, EncodeRequestOptions{Model: "claude-sonnet-4"})
		if err != nil {
			t.Fatalf("EncodeUpstreamRequest() error = %v", err)
		}
		var decoded map[string]any
		if err := json.Unmarshal(raw, &decoded); err != nil {
			t.Fatalf("Unmarshal() error = %v", err)
		}
		if decoded["thinking"] != nil {
			t.Fatalf("thinking = %+v, want it dropped", decoded["thinking"])
		}
		found := false
		for _, warning := range req.Warnings {
			if warning.Code == LossDroppedReasoning && warning.Path == "thinking" {
				found = true
			}
		}
		if !found {
			t.Fatalf("warnings = %+v, want a dropped-reasoning report", req.Warnings)
		}
	})

	t.Run("no room at all is reported", func(t *testing.T) {
		limit := minAnthropicThinkingBudgetTokens
		thinking, warnings := anthropicThinkingFor(t, `{"model":"gpt-5.4","reasoning_effort":"high","messages":[{"role":"user","content":"Hello"}]}`, &limit)
		if thinking != nil {
			t.Fatalf("thinking = %+v, want none when max_tokens leaves no room", thinking)
		}
		found := false
		for _, warning := range warnings {
			if warning.Code == LossDroppedReasoning {
				found = true
			}
		}
		if !found {
			t.Fatalf("warnings = %+v, want a report", warnings)
		}
	})
}

// The reverse direction: a budget becomes the nearest level. Chat documents up
// to "high", so the Responses-only "xhigh" reads back as high there.
func TestThinkingBudgetBecomesAReasoningLevel(t *testing.T) {
	cases := []struct {
		budget     int
		wantChat   string
		wantRespon string
	}{
		{1024, "medium", "medium"},
		{2048, "medium", "medium"},
		{4096, "high", "high"},
		{8192, "high", "xhigh"},
	}
	for _, tc := range cases {
		budget := tc.budget
		req := &LLMRequest{
			Protocol:              ProtocolAnthropicMessages,
			Model:                 "claude-sonnet-4",
			MaxOutputTokens:       intPtr(16384),
			Reasoning:             boolPtr(true),
			ReasoningBudgetTokens: &budget,
			Prompt:                []Message{{Role: RoleUser, Parts: []Part{{Type: PartText, Text: &TextPart{Text: "Hello"}}}}},
		}

		chatRaw, err := NewOpenAIChatAdapter().EncodeRequest(req, EncodeRequestOptions{Model: "gpt-5.4"})
		if err != nil {
			t.Fatalf("chat EncodeRequest() error = %v", err)
		}
		var chatDecoded map[string]any
		if err := json.Unmarshal(chatRaw, &chatDecoded); err != nil {
			t.Fatalf("Unmarshal() error = %v", err)
		}
		if got := chatDecoded["reasoning_effort"]; got != tc.wantChat {
			t.Fatalf("budget %d: chat reasoning_effort = %v, want %s", tc.budget, got, tc.wantChat)
		}

		responsesRaw, err := NewOpenAIResponsesAdapter().EncodeRequest(req, EncodeRequestOptions{Model: "gpt-5.4"})
		if err != nil {
			t.Fatalf("responses EncodeRequest() error = %v", err)
		}
		var responsesDecoded map[string]any
		if err := json.Unmarshal(responsesRaw, &responsesDecoded); err != nil {
			t.Fatalf("Unmarshal() error = %v", err)
		}
		reasoning, _ := responsesDecoded["reasoning"].(map[string]any)
		if got := reasoning["effort"]; got != tc.wantRespon {
			t.Fatalf("budget %d: responses effort = %v, want %s", tc.budget, got, tc.wantRespon)
		}
	}
}

func boolPtr(value bool) *bool { return &value }
