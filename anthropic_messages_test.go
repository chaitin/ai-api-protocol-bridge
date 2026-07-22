package protocolbridge

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestAnthropicMessagesDecodeRequest(t *testing.T) {
	adapter := NewAnthropicMessagesAdapter()
	raw := []byte(`{
		"model":"qwen3.5-plus",
		"system":[{"type":"text","text":"You are helpful.","cache_control":{"type":"ephemeral"}}],
		"max_tokens":64,
		"temperature":0.2,
		"output_config":{"format":{"type":"json_schema","schema":{"type":"object"},"strict":true}},
		"thinking":{"type":"enabled","budget_tokens":1024,"display":"summarized"},
		"metadata":{"user_id":"dev-user"},
		"messages":[{"role":"user","content":[{"type":"text","text":"Hello"}]}],
		"tools":[{"name":"get_weather","description":"Get weather.","input_schema":{"type":"object"},"strict":true}],
		"tool_choice":{"type":"any","disable_parallel_tool_use":true}
	}`)

	req, err := adapter.DecodeRequest(raw)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}

	if req.Model != "qwen3.5-plus" {
		t.Fatalf("Model = %q", req.Model)
	}
	if len(req.Prompt) != 2 {
		t.Fatalf("Prompt length = %d", len(req.Prompt))
	}
	if req.Prompt[0].Role != RoleSystem || req.Prompt[0].Parts[0].Text.Text != "You are helpful." {
		t.Fatalf("system message = %+v", req.Prompt[0])
	}
	if req.Prompt[1].Role != RoleUser || req.Prompt[1].Parts[0].Text.Text != "Hello" {
		t.Fatalf("user message = %+v", req.Prompt[1])
	}
	if req.MaxOutputTokens == nil || *req.MaxOutputTokens != 64 {
		t.Fatalf("MaxOutputTokens = %v", req.MaxOutputTokens)
	}
	if req.ResponseFormat == nil || req.ResponseFormat.Type != ResponseFormatJSON || req.ResponseFormat.Strict == nil || !*req.ResponseFormat.Strict {
		t.Fatalf("ResponseFormat = %+v", req.ResponseFormat)
	}
	if req.Cache == nil || !*req.Cache {
		t.Fatalf("Cache = %+v", req.Cache)
	}
	if len(req.Tools) != 1 || req.Tools[0].InputSchema["type"] != "object" || req.Tools[0].Strict == nil || !*req.Tools[0].Strict {
		t.Fatalf("Tools = %+v", req.Tools)
	}
	if req.ToolChoice == nil || req.ToolChoice.Type != ToolChoiceRequired {
		t.Fatalf("ToolChoice = %+v", req.ToolChoice)
	}
	if req.ParallelToolCalls == nil || *req.ParallelToolCalls {
		t.Fatalf("ParallelToolCalls = %+v", req.ParallelToolCalls)
	}
	if req.Reasoning == nil || !*req.Reasoning {
		t.Fatalf("Reasoning = %+v", req.Reasoning)
	}
	if req.ProviderOptions != nil {
		t.Fatalf("ProviderOptions = %+v", req.ProviderOptions)
	}
}

func TestAnthropicMessagesDecodeRequestFiltersClaudeCodeVolatileSystem(t *testing.T) {
	adapter := NewAnthropicMessagesAdapter()
	raw := []byte(`{
		"model":"claude",
		"system":[
			{"type":"text","text":"cc_version=2.1.122.d65; cc_entrypoint=cli; cch=00571;"},
			{"type":"text","text":"x-anthropic-billing-header: cc_version=2.1.122.d65; cc_entrypoint=cli; cch=00571;"},
			{"type":"text","text":"You are helpful."}
		],
		"messages":[{"role":"user","content":"Hello"}]
	}`)

	req, err := adapter.DecodeRequest(raw)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	if len(req.Prompt) != 2 {
		t.Fatalf("Prompt length = %d", len(req.Prompt))
	}
	if req.Prompt[0].Role != RoleSystem || req.Prompt[0].Parts[0].Text.Text != "You are helpful." {
		t.Fatalf("system message = %+v", req.Prompt[0])
	}

	bridge, ok := NewCrossFamilyBridge(ProtocolAnthropicMessages, "openai")
	if !ok {
		t.Fatal("NewCrossFamilyBridge() ok = false, want true")
	}
	responsesRaw, err := bridge.EncodeUpstreamRequest(req, EncodeRequestOptions{Model: "gpt-5.4"})
	if err != nil {
		t.Fatalf("EncodeUpstreamRequest() error = %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(responsesRaw, &decoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	if decoded["instructions"] != "You are helpful." {
		t.Fatalf("instructions = %v", decoded["instructions"])
	}
}

func TestAnthropicMessagesEncodeRequest(t *testing.T) {
	adapter := NewAnthropicMessagesAdapter()
	maxTokens := 64
	temperature := 0.2
	reasoning := true
	strict := true
	cache := true
	parallelToolCalls := false
	req := &LLMRequest{
		Model: "qwen3.5-plus",
		Prompt: []Message{
			{Role: RoleSystem, Parts: []Part{{Type: PartText, Text: &TextPart{Text: "You are helpful."}}}},
			{Role: RoleDeveloper, Parts: []Part{{Type: PartText, Text: &TextPart{Text: "Follow project rules."}}}},
			{Role: RoleUser, Parts: []Part{{Type: PartText, Text: &TextPart{Text: "Hello"}}}},
		},
		MaxOutputTokens:   &maxTokens,
		Temperature:       &temperature,
		ResponseFormat:    &ResponseFormat{Type: ResponseFormatJSON, Schema: map[string]any{"type": "object"}, Strict: &strict},
		Cache:             &cache,
		Reasoning:         &reasoning,
		Tools:             []Tool{{Type: ToolFunction, Name: "get_weather", Description: "Get weather.", InputSchema: map[string]any{"type": "object"}, Strict: &strict}},
		ToolChoice:        &ToolChoice{Type: ToolChoiceRequired},
		ParallelToolCalls: &parallelToolCalls,
	}

	raw, err := adapter.EncodeRequest(req, EncodeRequestOptions{Model: "upstream-qwen"})
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}

	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	if decoded["model"] != "upstream-qwen" {
		t.Fatalf("request = %+v", decoded)
	}
	system := decoded["system"].([]any)
	systemBlock := system[0].(map[string]any)
	if systemBlock["text"] != "You are helpful.\nFollow project rules." {
		t.Fatalf("system = %+v", system)
	}
	if decoded["max_tokens"] != float64(64) {
		t.Fatalf("max_tokens = %v", decoded["max_tokens"])
	}
	if _, ok := decoded["cache_control"]; ok {
		t.Fatalf("top-level cache_control should be omitted: %+v", decoded)
	}
	cacheControl := systemBlock["cache_control"].(map[string]any)
	if cacheControl["type"] != "ephemeral" {
		t.Fatalf("cache_control = %+v", cacheControl)
	}
	outputConfig := decoded["output_config"].(map[string]any)
	format := outputConfig["format"].(map[string]any)
	if format["type"] != "json_schema" || format["strict"] != true {
		t.Fatalf("output_config = %+v", outputConfig)
	}
	toolChoice := decoded["tool_choice"].(map[string]any)
	if toolChoice["type"] != "any" || toolChoice["disable_parallel_tool_use"] != true {
		t.Fatalf("tool_choice = %+v", toolChoice)
	}
	if _, ok := decoded["thinking"]; ok {
		t.Fatalf("thinking should be omitted when max_tokens cannot fit a valid budget: %+v", decoded["thinking"])
	}
	if _, ok := decoded["service_tier"]; ok {
		t.Fatalf("service_tier should be omitted: %+v", decoded)
	}
	messages := decoded["messages"].([]any)
	if len(messages) != 1 || messages[0].(map[string]any)["role"] != "user" {
		t.Fatalf("messages = %+v", messages)
	}
	tool := decoded["tools"].([]any)[0].(map[string]any)
	if tool["strict"] != true {
		t.Fatalf("tool = %+v", tool)
	}
}

func TestAnthropicMessagesEncodeRequestDowngradesForcedToolChoiceWithThinking(t *testing.T) {
	adapter := NewAnthropicMessagesAdapter()
	reasoning := true
	req := &LLMRequest{
		Model: "qwen3.5-plus",
		Prompt: []Message{
			{Role: RoleUser, Parts: []Part{{Type: PartText, Text: &TextPart{Text: "Hello"}}}},
		},
		Reasoning:  &reasoning,
		Tools:      []Tool{{Type: ToolFunction, Name: "get_weather", InputSchema: map[string]any{"type": "object"}}},
		ToolChoice: &ToolChoice{Type: ToolChoiceTool, ToolName: "get_weather"},
	}

	raw, err := adapter.EncodeRequest(req, EncodeRequestOptions{Model: "upstream-qwen"})
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}

	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	toolChoice := decoded["tool_choice"].(map[string]any)
	if toolChoice["type"] != "auto" {
		t.Fatalf("tool_choice = %+v", toolChoice)
	}
	if _, ok := toolChoice["name"]; ok {
		t.Fatalf("tool_choice name should be omitted: %+v", toolChoice)
	}
}

func TestAnthropicMessagesEncodeToolResultAsUserMessage(t *testing.T) {
	adapter := NewAnthropicMessagesAdapter()
	req := &LLMRequest{
		Model: "claude-sonnet",
		Prompt: []Message{
			{Role: RoleTool, Parts: []Part{{Type: PartToolResult, ToolResult: &ToolResultPart{ToolCallID: "toolu_1", Output: ToolResultOutput{Type: ToolResultText, Text: "Sunny"}}}}},
		},
	}

	raw, err := adapter.EncodeRequest(req, EncodeRequestOptions{Model: "upstream-claude"})
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}

	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	messages := decoded["messages"].([]any)
	message := messages[0].(map[string]any)
	if message["role"] != "user" {
		t.Fatalf("tool result role = %v", message["role"])
	}
	content := message["content"].([]any)[0].(map[string]any)
	if content["type"] != "tool_result" || content["tool_use_id"] != "toolu_1" || content["content"] != "Sunny" {
		t.Fatalf("tool result content = %+v", content)
	}
	if decoded["max_tokens"] != float64(defaultMaxOutputTokens) {
		t.Fatalf("max_tokens = %v", decoded["max_tokens"])
	}
}

func TestAnthropicMessagesDecodeBlockCacheControl(t *testing.T) {
	adapter := NewAnthropicMessagesAdapter()
	req, err := adapter.DecodeRequest([]byte(`{
		"model":"claude",
		"messages":[{"role":"user","content":[{"type":"text","text":"Hello","cache_control":{"type":"ephemeral"}}]}]
	}`))
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	if req.Cache == nil || !*req.Cache {
		t.Fatalf("Cache = %+v", req.Cache)
	}
}

func TestAnthropicMessagesEncodeRequestDefaultsCacheOn(t *testing.T) {
	adapter := NewAnthropicMessagesAdapter()
	req := &LLMRequest{
		Model:  "claude",
		Prompt: []Message{{Role: RoleUser, Parts: []Part{{Type: PartText, Text: &TextPart{Text: "Hello"}}}}},
	}

	raw, err := adapter.EncodeRequest(req, EncodeRequestOptions{Model: "claude"})
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	if _, ok := decoded["cache_control"]; ok {
		t.Fatalf("top-level cache_control should be omitted: %+v", decoded)
	}
	messages := decoded["messages"].([]any)
	content := messages[0].(map[string]any)["content"].([]any)
	cacheControl := content[0].(map[string]any)["cache_control"].(map[string]any)
	if cacheControl["type"] != "ephemeral" {
		t.Fatalf("cache_control = %+v", cacheControl)
	}
}

func TestAnthropicMessagesDecodeTopLevelCacheControlIgnored(t *testing.T) {
	adapter := NewAnthropicMessagesAdapter()
	req, err := adapter.DecodeRequest([]byte(`{
		"model":"claude",
		"cache_control":{"type":"ephemeral"},
		"messages":[{"role":"user","content":"Hello"}]
	}`))
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	if req.Cache != nil {
		t.Fatalf("Cache = %+v", req.Cache)
	}
}

func TestAnthropicMessagesEncodeRequestCacheFalse(t *testing.T) {
	adapter := NewAnthropicMessagesAdapter()
	cache := false
	req := &LLMRequest{
		Model:  "claude",
		Prompt: []Message{{Role: RoleUser, Parts: []Part{{Type: PartText, Text: &TextPart{Text: "Hello"}}}}},
		Cache:  &cache,
	}

	raw, err := adapter.EncodeRequest(req, EncodeRequestOptions{Model: "claude"})
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	if _, ok := decoded["cache_control"]; ok {
		t.Fatalf("cache_control should be omitted: %+v", decoded)
	}
}

func TestAnthropicMessagesDecodeRequestDefaults(t *testing.T) {
	adapter := NewAnthropicMessagesAdapter()
	raw := []byte(`{"model":"claude","messages":[{"role":"user","content":"Hello"}]}`)

	req, err := adapter.DecodeRequest(raw)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	if req.MaxOutputTokens == nil || *req.MaxOutputTokens != defaultMaxOutputTokens {
		t.Fatalf("MaxOutputTokens = %v", req.MaxOutputTokens)
	}
}

func TestAnthropicMessagesThinkingBudgetOverride(t *testing.T) {
	adapter := NewAnthropicMessagesAdapter()
	req, err := adapter.DecodeRequest([]byte(`{
		"model":"claude",
		"max_tokens":4096,
		"thinking":{"type":"enabled","budget_tokens":1024},
		"messages":[{"role":"user","content":"Hello"}]
	}`))
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	if req.ReasoningBudgetTokens == nil || *req.ReasoningBudgetTokens != 1024 {
		t.Fatalf("ReasoningBudgetTokens = %v", req.ReasoningBudgetTokens)
	}

	raw, err := adapter.EncodeRequest(req, EncodeRequestOptions{Model: "claude"})
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	thinking := decoded["thinking"].(map[string]any)
	if thinking["budget_tokens"] != float64(1024) {
		t.Fatalf("thinking = %+v", thinking)
	}
}

func TestAnthropicMessagesDecodeAdaptiveThinkingIgnoresOutputEffort(t *testing.T) {
	adapter := NewAnthropicMessagesAdapter()
	req, err := adapter.DecodeRequest([]byte(`{
		"model":"claude",
		"max_tokens":2048,
		"thinking":{"type":"adaptive"},
		"output_config":{"effort":"max"},
		"metadata":{"user_id":"session-1"},
		"messages":[{"role":"user","content":"Hello"}]
	}`))
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	if req.ReasoningEffort != "high" {
		t.Fatalf("ReasoningEffort = %q, want high", req.ReasoningEffort)
	}
	if req.Reasoning != nil && *req.Reasoning {
		t.Fatalf("Reasoning = %v, want not enabled by bool", req.Reasoning)
	}
	if req.Metadata["user_id"] != "session-1" {
		t.Fatalf("Metadata = %+v", req.Metadata)
	}
}

func TestAnthropicMessagesDecodeFiltersNativeTools(t *testing.T) {
	adapter := NewAnthropicMessagesAdapter()
	req, err := adapter.DecodeRequest([]byte(`{
		"model":"claude",
		"max_tokens":128,
		"tools":[
			{"type":"web_search_20250305","name":"web_search"},
			{"name":"regular","input_schema":{"type":"object"}}
		],
		"messages":[{"role":"user","content":"Hello"}]
	}`))
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	if len(req.Tools) != 1 || req.Tools[0].Name != "regular" {
		t.Fatalf("Tools = %+v", req.Tools)
	}
}

func TestAnthropicMessagesThinkingBudgetDefaultsToSafeBudget(t *testing.T) {
	adapter := NewAnthropicMessagesAdapter()
	maxTokens := 4096
	reasoning := true
	req := &LLMRequest{
		Model:           "claude",
		Prompt:          []Message{{Role: RoleUser, Parts: []Part{{Type: PartText, Text: &TextPart{Text: "Hello"}}}}},
		MaxOutputTokens: &maxTokens,
		Reasoning:       &reasoning,
	}

	raw, err := adapter.EncodeRequest(req, EncodeRequestOptions{Model: "claude"})
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	thinking := decoded["thinking"].(map[string]any)
	if thinking["budget_tokens"] != float64(defaultThinkingBudgetTokens) {
		t.Fatalf("thinking = %+v", thinking)
	}
}

func TestAnthropicMessagesOmitThinkingWhenMaxTokensTooSmall(t *testing.T) {
	adapter := NewAnthropicMessagesAdapter()
	maxTokens := 128
	reasoning := true
	req := &LLMRequest{
		Model:           "claude",
		Prompt:          []Message{{Role: RoleUser, Parts: []Part{{Type: PartText, Text: &TextPart{Text: "Hello"}}}}},
		MaxOutputTokens: &maxTokens,
		Reasoning:       &reasoning,
	}

	raw, err := adapter.EncodeRequest(req, EncodeRequestOptions{Model: "claude"})
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	if _, ok := decoded["thinking"]; ok {
		t.Fatalf("thinking should be omitted: %+v", decoded["thinking"])
	}
}

func TestAnthropicMessagesDecodeResponse(t *testing.T) {
	adapter := NewAnthropicMessagesAdapter()
	raw := []byte(`{
		"id":"msg_1",
		"type":"message",
		"role":"assistant",
		"model":"qwen3.5-plus",
		"content":[{"type":"thinking","thinking":"I should answer briefly.","signature":"sig_1"},{"type":"redacted_thinking","data":"encrypted","signature":"sig_2"},{"type":"text","text":"Hello back"},{"type":"tool_result","tool_use_id":"toolu_1","content":"failed","is_error":true}],
		"stop_reason":"end_turn",
		"usage":{"input_tokens":10,"output_tokens":5,"cache_creation_input_tokens":3,"cache_read_input_tokens":4}
	}`)

	resp, err := adapter.DecodeResponse(raw)
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	if resp.ID != "msg_1" || resp.Model != "qwen3.5-plus" {
		t.Fatalf("response = %+v", resp)
	}
	if resp.Content[0].Reasoning.Text != "I should answer briefly." || resp.Content[0].Reasoning.Signature != "sig_1" {
		t.Fatalf("thinking = %+v", resp.Content[0])
	}
	if resp.Content[1].Reasoning.Redacted != "encrypted" || resp.Content[1].Reasoning.Signature != "sig_2" {
		t.Fatalf("redacted thinking = %+v", resp.Content[1])
	}
	if resp.Content[2].Text.Text != "Hello back" || resp.Content[3].ToolResult.Output.Type != ToolResultErrorText {
		t.Fatalf("content = %+v", resp.Content)
	}
	if resp.FinishReason != FinishStop {
		t.Fatalf("FinishReason = %q", resp.FinishReason)
	}
	if *resp.Usage.InputTokens != 10 || *resp.Usage.OutputTokens != 5 || *resp.Usage.CacheCreationInputTokens != 3 || *resp.Usage.CacheReadInputTokens != 4 || *resp.Usage.CachedInputTokens != 4 {
		t.Fatalf("usage = %+v", resp.Usage)
	}
	billingUsage := resp.BillingUsage()
	if billingUsage.InputTokens != 13 || billingUsage.CachedInputTokens != 4 || billingUsage.OutputTokens != 5 {
		t.Fatalf("billing usage = %+v", billingUsage)
	}
}

func TestAnthropicMessagesEncodeResponse(t *testing.T) {
	adapter := NewAnthropicMessagesAdapter()
	inputTokens := 10
	cachedInputTokens := 3
	outputTokens := 5
	resp := &LLMResponse{
		Protocol: ProtocolAnthropicMessages,
		ID:       "msg_1",
		Model:    "qwen3.5-plus",
		Role:     RoleAssistant,
		Content: []Part{
			{Type: PartReasoning, Reasoning: &ReasoningPart{Text: "I should answer briefly.", Signature: "sig_1"}},
			{Type: PartReasoning, Reasoning: &ReasoningPart{Redacted: "encrypted", Signature: "sig_2"}},
			{Type: PartText, Text: &TextPart{Text: "Hello back"}},
			{Type: PartFile, File: &FilePart{Type: FileImage, MediaType: "image/png", Data: "abc"}},
			{Type: PartToolResult, ToolResult: &ToolResultPart{ToolCallID: "toolu_1", Output: ToolResultOutput{Type: ToolResultErrorText, Text: "failed"}}},
			{Type: PartToolResult, ToolResult: &ToolResultPart{ToolCallID: "toolu_2", Output: ToolResultOutput{Type: ToolResultContent, Content: []Part{{Type: PartText, Text: &TextPart{Text: "structured"}}}}}},
		},
		FinishReason: FinishStop,
		Usage:        Usage{InputTokens: &inputTokens, OutputTokens: &outputTokens, CacheReadInputTokens: &cachedInputTokens},
	}

	raw, err := adapter.EncodeResponse(resp, EncodeResponseOptions{Model: "qwen3.5-plus"})
	if err != nil {
		t.Fatalf("EncodeResponse() error = %v", err)
	}

	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	if decoded["type"] != "message" || decoded["role"] != "assistant" || decoded["model"] != "qwen3.5-plus" {
		t.Fatalf("response = %+v", decoded)
	}
	content := decoded["content"].([]any)
	reasoning := content[0].(map[string]any)
	redacted := content[1].(map[string]any)
	text := content[2].(map[string]any)
	image := content[3].(map[string]any)
	toolError := content[4].(map[string]any)
	toolContent := content[5].(map[string]any)
	if reasoning["type"] != "thinking" || reasoning["thinking"] != "I should answer briefly." || reasoning["signature"] != "sig_1" {
		t.Fatalf("reasoning = %+v", reasoning)
	}
	if redacted["type"] != "redacted_thinking" || redacted["data"] != "encrypted" {
		t.Fatalf("redacted = %+v", redacted)
	}
	if _, ok := redacted["signature"]; ok {
		t.Fatalf("redacted = %+v", redacted)
	}
	if text["type"] != "text" || text["text"] != "Hello back" {
		t.Fatalf("text = %+v", text)
	}
	if image["type"] != "image" {
		t.Fatalf("image = %+v", image)
	}
	if toolError["type"] != "tool_result" || toolError["is_error"] != true || toolError["content"] != "failed" {
		t.Fatalf("tool error = %+v", toolError)
	}
	structuredContent := toolContent["content"].([]any)[0].(map[string]any)
	if structuredContent["type"] != "text" || structuredContent["text"] != "structured" {
		t.Fatalf("tool content = %+v", toolContent)
	}
	usage := decoded["usage"].(map[string]any)
	if usage["input_tokens"] != float64(10) || usage["cache_read_input_tokens"] != float64(3) || usage["output_tokens"] != float64(5) {
		t.Fatalf("usage = %+v", usage)
	}
}

func TestAnthropicMessagesEncodeResponseUsesFirstChoice(t *testing.T) {
	adapter := NewAnthropicMessagesAdapter()
	resp := &LLMResponse{
		Protocol: ProtocolOpenAIChat,
		ID:       "chatcmpl_1",
		Model:    "gpt-4.1",
		Choices: []LLMChoice{
			{Index: 0, Role: RoleAssistant, Content: []Part{{Type: PartText, Text: &TextPart{Text: "First"}}}, FinishReason: FinishStop},
			{Index: 1, Role: RoleAssistant, Content: []Part{{Type: PartText, Text: &TextPart{Text: "Second"}}}, FinishReason: FinishLength},
		},
		FinishReason: FinishUnknown,
	}

	raw, err := adapter.EncodeResponse(resp, EncodeResponseOptions{Model: "claude"})
	if err != nil {
		t.Fatalf("EncodeResponse() error = %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	content := decoded["content"].([]any)
	if content[0].(map[string]any)["text"] != "First" || decoded["stop_reason"] != "end_turn" {
		t.Fatalf("response = %+v", decoded)
	}
}

func TestAnthropicMessagesEncodeContentFilterAsRefusal(t *testing.T) {
	adapter := NewAnthropicMessagesAdapter()
	resp := &LLMResponse{
		Protocol:     ProtocolOpenAIResponses,
		ID:           "resp_1",
		Model:        "gpt-5.4",
		Content:      []Part{{Type: PartRefusal, Refusal: &RefusalPart{Text: "I'm sorry, I cannot assist with that request."}}},
		FinishReason: FinishContentFilter,
	}

	raw, err := adapter.EncodeResponse(resp, EncodeResponseOptions{Model: "claude"})
	if err != nil {
		t.Fatalf("EncodeResponse() error = %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	if decoded["stop_reason"] != "refusal" {
		t.Fatalf("stop_reason = %v", decoded["stop_reason"])
	}
	stopDetails := decoded["stop_details"].(map[string]any)
	if stopDetails["type"] != "refusal" || stopDetails["explanation"] != "I'm sorry, I cannot assist with that request." {
		t.Fatalf("stop_details = %+v", stopDetails)
	}
}

func TestAnthropicMessagesStreamDecoder(t *testing.T) {
	decoder, err := NewAnthropicMessagesAdapter().NewStreamDecoder(StreamDecodeOptions{})
	if err != nil {
		t.Fatalf("NewStreamDecoder() error = %v", err)
	}

	parts, err := decoder.Decode(RawStreamEvent{Event: "message_start", Data: []byte(`{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude","usage":{"input_tokens":10,"cache_read_input_tokens":3}}}`)})
	if err != nil {
		t.Fatalf("Decode(message_start) error = %v", err)
	}
	if len(parts) != 1 || parts[0].Type != StreamStart || parts[0].ID != "msg_1" || parts[0].Usage.InputTokens == nil || *parts[0].Usage.InputTokens != 10 {
		t.Fatalf("message_start parts = %+v", parts)
	}

	parts, err = decoder.Decode(RawStreamEvent{Event: "content_block_start", Data: []byte(`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`)})
	if err != nil {
		t.Fatalf("Decode(content_block_start) error = %v", err)
	}
	if len(parts) != 1 || parts[0].Type != StreamTextStart || parts[0].ID != "content_0" {
		t.Fatalf("content_block_start parts = %+v", parts)
	}

	parts, err = decoder.Decode(RawStreamEvent{Event: "content_block_delta", Data: []byte(`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}`)})
	if err != nil {
		t.Fatalf("Decode(content_block_delta) error = %v", err)
	}
	if len(parts) != 1 || parts[0].Type != StreamTextDelta || parts[0].Delta != "Hi" {
		t.Fatalf("content_block_delta parts = %+v", parts)
	}

	parts, err = decoder.Decode(RawStreamEvent{Event: "content_block_stop", Data: []byte(`{"type":"content_block_stop","index":0}`)})
	if err != nil {
		t.Fatalf("Decode(content_block_stop) error = %v", err)
	}
	if len(parts) != 1 || parts[0].Type != StreamTextEnd {
		t.Fatalf("content_block_stop parts = %+v", parts)
	}

	parts, err = decoder.Decode(RawStreamEvent{Event: "message_delta", Data: []byte(`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}`)})
	if err != nil {
		t.Fatalf("Decode(message_delta) error = %v", err)
	}
	if len(parts) != 1 || parts[0].Type != StreamFinish || parts[0].FinishReason != FinishStop || parts[0].Usage.OutputTokens == nil || *parts[0].Usage.OutputTokens != 5 {
		t.Fatalf("message_delta parts = %+v", parts)
	}
	if parts[0].Usage.InputTokens == nil || *parts[0].Usage.InputTokens != 10 || parts[0].Usage.CacheReadInputTokens == nil || *parts[0].Usage.CacheReadInputTokens != 3 {
		t.Fatalf("message_delta merged usage = %+v", parts[0].Usage)
	}
}

func TestAnthropicMessagesStreamEncoder(t *testing.T) {
	encoder, err := NewAnthropicMessagesAdapter().NewStreamEncoder(StreamEncodeOptions{Model: "claude"})
	if err != nil {
		t.Fatalf("NewStreamEncoder() error = %v", err)
	}
	inputTokens := 10
	outputTokens := 5

	events, err := encoder.Encode(StreamPart{Type: StreamStart, ID: "msg_1", Usage: Usage{InputTokens: &inputTokens}})
	if err != nil {
		t.Fatalf("Encode(StreamStart) error = %v", err)
	}
	assertAnthropicStreamEvent(t, events, "message_start", "message_start")
	var startMessage anthropicStreamEvent
	if err := json.Unmarshal(events[0].Data, &startMessage); err != nil {
		t.Fatalf("Unmarshal(message_start) error = %v", err)
	}
	if startMessage.Message == nil || len(startMessage.Message.Content) != 0 {
		t.Fatalf("message_start content = %+v", startMessage.Message)
	}
	var rawStartMessage map[string]any
	if err := json.Unmarshal(events[0].Data, &rawStartMessage); err != nil {
		t.Fatalf("Unmarshal(raw message_start) error = %v", err)
	}
	messageMap := rawStartMessage["message"].(map[string]any)
	if _, ok := messageMap["stop_sequence"]; !ok {
		t.Fatalf("message_start should include stop_sequence: %+v", messageMap)
	}
	if _, ok := messageMap["stop_reason"]; !ok {
		t.Fatalf("message_start should include stop_reason: %+v", messageMap)
	}

	events, err = encoder.Encode(StreamPart{Type: StreamTextStart, ID: "text_1"})
	if err != nil {
		t.Fatalf("Encode(StreamTextStart) error = %v", err)
	}
	assertAnthropicStreamEvent(t, events, "content_block_start", "content_block_start")

	events, err = encoder.Encode(StreamPart{Type: StreamTextDelta, ID: "text_1", Delta: "Hi"})
	if err != nil {
		t.Fatalf("Encode(StreamTextDelta) error = %v", err)
	}
	var decoded anthropicStreamEvent
	if err := json.Unmarshal(events[0].Data, &decoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	if decoded.Type != "content_block_delta" || decoded.Delta.Type != "text_delta" || decoded.Delta.Text != "Hi" {
		t.Fatalf("delta event = %+v", decoded)
	}
	if decoded.Index == nil || *decoded.Index != 0 {
		t.Fatalf("text delta index = %+v", decoded.Index)
	}

	events, err = encoder.Encode(StreamPart{Type: StreamTextEnd, ID: "text_1"})
	if err != nil {
		t.Fatalf("Encode(StreamTextEnd) error = %v", err)
	}
	assertAnthropicStreamEvent(t, events, "content_block_stop", "content_block_stop")

	events, err = encoder.Encode(StreamPart{Type: StreamFinish, FinishReason: FinishStop, Usage: Usage{OutputTokens: &outputTokens}})
	if err != nil {
		t.Fatalf("Encode(StreamFinish) error = %v", err)
	}
	if len(events) != 2 || events[0].Event != "message_delta" || events[1].Event != "message_stop" {
		t.Fatalf("finish events = %+v", events)
	}
	var rawMessageDelta map[string]any
	if err := json.Unmarshal(events[0].Data, &rawMessageDelta); err != nil {
		t.Fatalf("Unmarshal(raw message_delta) error = %v", err)
	}
	deltaMap := rawMessageDelta["delta"].(map[string]any)
	if _, ok := deltaMap["stop_sequence"]; !ok {
		t.Fatalf("message_delta should include stop_sequence: %+v", deltaMap)
	}
}

func TestAnthropicMessagesStreamEncoderFinishDefaultsOutputTokens(t *testing.T) {
	encoder, err := NewAnthropicMessagesAdapter().NewStreamEncoder(StreamEncodeOptions{Model: "claude"})
	if err != nil {
		t.Fatalf("NewStreamEncoder() error = %v", err)
	}

	events, err := encoder.Encode(StreamPart{Type: StreamFinish, FinishReason: FinishStop})
	if err != nil {
		t.Fatalf("Encode(StreamFinish) error = %v", err)
	}
	if len(events) != 2 || events[0].Event != "message_delta" {
		t.Fatalf("finish events = %+v", events)
	}
	var decoded anthropicStreamEvent
	if err := json.Unmarshal(events[0].Data, &decoded); err != nil {
		t.Fatalf("Unmarshal(message_delta) error = %v", err)
	}
	if decoded.Usage == nil || decoded.Usage.OutputTokens == nil || *decoded.Usage.OutputTokens != 0 {
		t.Fatalf("message_delta usage = %+v", decoded.Usage)
	}
}

func TestAnthropicMessagesStreamEncoderReusesSingleActiveBlockForEmptyDeltaID(t *testing.T) {
	encoder, err := NewAnthropicMessagesAdapter().NewStreamEncoder(StreamEncodeOptions{Model: "claude"})
	if err != nil {
		t.Fatalf("NewStreamEncoder() error = %v", err)
	}

	startEvents, err := encoder.Encode(StreamPart{Type: StreamTextStart, ID: "text_1"})
	if err != nil {
		t.Fatalf("Encode(StreamTextStart) error = %v", err)
	}
	deltaEvents, err := encoder.Encode(StreamPart{Type: StreamTextDelta, Delta: "Hi"})
	if err != nil {
		t.Fatalf("Encode(StreamTextDelta) error = %v", err)
	}
	moreDeltaEvents, err := encoder.Encode(StreamPart{Type: StreamTextDelta, Delta: "!"})
	if err != nil {
		t.Fatalf("Encode(StreamTextDelta 2) error = %v", err)
	}
	stopEvents, err := encoder.Encode(StreamPart{Type: StreamTextEnd})
	if err != nil {
		t.Fatalf("Encode(StreamTextEnd) error = %v", err)
	}

	var start, delta, moreDelta, stop anthropicStreamEvent
	if err := json.Unmarshal(startEvents[0].Data, &start); err != nil {
		t.Fatalf("Unmarshal(start) error = %v", err)
	}
	if err := json.Unmarshal(deltaEvents[0].Data, &delta); err != nil {
		t.Fatalf("Unmarshal(delta) error = %v", err)
	}
	if err := json.Unmarshal(moreDeltaEvents[0].Data, &moreDelta); err != nil {
		t.Fatalf("Unmarshal(more delta) error = %v", err)
	}
	if err := json.Unmarshal(stopEvents[0].Data, &stop); err != nil {
		t.Fatalf("Unmarshal(stop) error = %v", err)
	}
	if start.Index == nil || delta.Index == nil || moreDelta.Index == nil || stop.Index == nil {
		t.Fatalf("expected indexes: start=%+v delta=%+v more=%+v stop=%+v", start, delta, moreDelta, stop)
	}
	if *start.Index != 0 || *delta.Index != 0 || *moreDelta.Index != 0 || *stop.Index != 0 {
		t.Fatalf("indexes mismatch: start=%d delta=%d more=%d stop=%d", *start.Index, *delta.Index, *moreDelta.Index, *stop.Index)
	}
}

func TestAnthropicMessagesStreamEncoderIgnoresEmptyEndWithoutActiveBlock(t *testing.T) {
	encoder, err := NewAnthropicMessagesAdapter().NewStreamEncoder(StreamEncodeOptions{Model: "claude"})
	if err != nil {
		t.Fatalf("NewStreamEncoder() error = %v", err)
	}

	for _, part := range []StreamPart{
		{Type: StreamTextEnd},
		{Type: StreamReasoningEnd},
		{Type: StreamToolInputEnd},
	} {
		events, err := encoder.Encode(part)
		if err != nil {
			t.Fatalf("Encode(%s) error = %v", part.Type, err)
		}
		if len(events) != 0 {
			t.Fatalf("Encode(%s) events = %+v", part.Type, events)
		}
	}
}

func TestAnthropicMessagesStreamEncoderIgnoresEndForUnknownID(t *testing.T) {
	encoder, err := NewAnthropicMessagesAdapter().NewStreamEncoder(StreamEncodeOptions{Model: "claude"})
	if err != nil {
		t.Fatalf("NewStreamEncoder() error = %v", err)
	}

	if _, err := encoder.Encode(StreamPart{Type: StreamToolInputStart, ID: "fc_1", ToolCallID: "call_1", ToolName: "get_weather"}); err != nil {
		t.Fatalf("Encode(StreamToolInputStart) error = %v", err)
	}
	stopEvents, err := encoder.Encode(StreamPart{Type: StreamToolInputEnd, ID: "fc_1", ToolCallID: "call_1"})
	if err != nil {
		t.Fatalf("Encode(StreamToolInputEnd) error = %v", err)
	}
	if len(stopEvents) != 1 {
		t.Fatalf("first stop events = %+v", stopEvents)
	}
	duplicateStopEvents, err := encoder.Encode(StreamPart{Type: StreamToolInputEnd, ID: "fc_1", ToolCallID: "call_1"})
	if err != nil {
		t.Fatalf("Encode(duplicate StreamToolInputEnd) error = %v", err)
	}
	if len(duplicateStopEvents) != 0 {
		t.Fatalf("duplicate stop events = %+v", duplicateStopEvents)
	}
	unknownStopEvents, err := encoder.Encode(StreamPart{Type: StreamTextEnd, ID: "unknown_text"})
	if err != nil {
		t.Fatalf("Encode(unknown StreamTextEnd) error = %v", err)
	}
	if len(unknownStopEvents) != 0 {
		t.Fatalf("unknown stop events = %+v", unknownStopEvents)
	}
}

func TestAnthropicMessagesStreamEncoderThinkingBlockMatchesAnthropicStyle(t *testing.T) {
	encoder, err := NewAnthropicMessagesAdapter().NewStreamEncoder(StreamEncodeOptions{Model: "claude"})
	if err != nil {
		t.Fatalf("NewStreamEncoder() error = %v", err)
	}

	events, err := encoder.Encode(StreamPart{Type: StreamReasoningStart, ID: "reason_1"})
	if err != nil {
		t.Fatalf("Encode(StreamReasoningStart) error = %v", err)
	}
	var decoded anthropicStreamEvent
	if err := json.Unmarshal(events[0].Data, &decoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	if decoded.Index == nil || *decoded.Index != 0 {
		t.Fatalf("thinking start index = %+v", decoded.Index)
	}
	if decoded.ContentBlock == nil || decoded.ContentBlock.Type != "thinking" || decoded.ContentBlock.Thinking != "" || decoded.ContentBlock.Signature != "" {
		t.Fatalf("thinking start block = %+v", decoded.ContentBlock)
	}
}

func TestAnthropicMessagesStreamEncoderToolIndexesAndOmitEmptyFields(t *testing.T) {
	encoder, err := NewAnthropicMessagesAdapter().NewStreamEncoder(StreamEncodeOptions{Model: "claude"})
	if err != nil {
		t.Fatalf("NewStreamEncoder() error = %v", err)
	}

	startEvents, err := encoder.Encode(StreamPart{Type: StreamToolInputStart, ToolCallID: "tool_1", ToolName: "Write"})
	if err != nil {
		t.Fatalf("Encode(StreamToolInputStart) error = %v", err)
	}
	deltaEvents, err := encoder.Encode(StreamPart{Type: StreamToolInputDelta, ToolCallID: "tool_1", Delta: `{"file_path":"/tmp/hello.c"}`})
	if err != nil {
		t.Fatalf("Encode(StreamToolInputDelta) error = %v", err)
	}
	stopEvents, err := encoder.Encode(StreamPart{Type: StreamToolInputEnd, ToolCallID: "tool_1"})
	if err != nil {
		t.Fatalf("Encode(StreamToolInputEnd) error = %v", err)
	}

	var start anthropicStreamEvent
	if err := json.Unmarshal(startEvents[0].Data, &start); err != nil {
		t.Fatalf("Unmarshal(start) error = %v", err)
	}
	var delta anthropicStreamEvent
	if err := json.Unmarshal(deltaEvents[0].Data, &delta); err != nil {
		t.Fatalf("Unmarshal(delta) error = %v", err)
	}
	var stop anthropicStreamEvent
	if err := json.Unmarshal(stopEvents[0].Data, &stop); err != nil {
		t.Fatalf("Unmarshal(stop) error = %v", err)
	}

	if start.Index == nil || delta.Index == nil || stop.Index == nil {
		t.Fatalf("expected indexes to be present: start=%+v delta=%+v stop=%+v", start, delta, stop)
	}
	if *start.Index != *delta.Index || *delta.Index != *stop.Index {
		t.Fatalf("tool indexes mismatch: start=%d delta=%d stop=%d", *start.Index, *delta.Index, *stop.Index)
	}
	if start.ContentBlock == nil || start.ContentBlock.Type != "tool_use" {
		t.Fatalf("start content_block = %+v", start.ContentBlock)
	}
	if delta.Delta == nil || delta.Delta.Type != "input_json_delta" || delta.Delta.PartialJSON == "" {
		t.Fatalf("delta event = %+v", delta)
	}
	if start.Delta != nil || start.Usage != nil {
		t.Fatalf("start should omit empty delta/usage: %+v", start)
	}
	if delta.ContentBlock != nil || delta.Usage != nil {
		t.Fatalf("delta should omit empty content_block/usage: %+v", delta)
	}
	if stop.ContentBlock != nil || stop.Delta != nil || stop.Usage != nil {
		t.Fatalf("stop should omit empty payload fields: %+v", stop)
	}
}

func TestAnthropicMessagesStreamDecodeEncodePreservesToolIndex(t *testing.T) {
	decoder, err := NewAnthropicMessagesAdapter().NewStreamDecoder(StreamDecodeOptions{})
	if err != nil {
		t.Fatalf("NewStreamDecoder() error = %v", err)
	}
	encoder, err := NewAnthropicMessagesAdapter().NewStreamEncoder(StreamEncodeOptions{Model: "claude"})
	if err != nil {
		t.Fatalf("NewStreamEncoder() error = %v", err)
	}

	startParts, err := decoder.Decode(RawStreamEvent{Event: "content_block_start", Data: []byte(`{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"tool_1","name":"Write","input":{}}}`)})
	if err != nil {
		t.Fatalf("Decode(start) error = %v", err)
	}
	deltaParts, err := decoder.Decode(RawStreamEvent{Event: "content_block_delta", Data: []byte(`{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"file_path\":\"/tmp/hello.c\"}"}}`)})
	if err != nil {
		t.Fatalf("Decode(delta) error = %v", err)
	}

	startEvents, err := encoder.Encode(startParts[0])
	if err != nil {
		t.Fatalf("Encode(start) error = %v", err)
	}
	deltaEvents, err := encoder.Encode(deltaParts[0])
	if err != nil {
		t.Fatalf("Encode(delta) error = %v", err)
	}

	var start anthropicStreamEvent
	if err := json.Unmarshal(startEvents[0].Data, &start); err != nil {
		t.Fatalf("Unmarshal(start) error = %v", err)
	}
	var delta anthropicStreamEvent
	if err := json.Unmarshal(deltaEvents[0].Data, &delta); err != nil {
		t.Fatalf("Unmarshal(delta) error = %v", err)
	}

	if start.Index == nil || delta.Index == nil {
		t.Fatalf("expected indexes to be present: start=%+v delta=%+v", start, delta)
	}
	if *start.Index != 0 || *delta.Index != 0 {
		t.Fatalf("re-encoded tool stream should reuse same encoder index: start=%d delta=%d", *start.Index, *delta.Index)
	}
	if delta.Delta == nil || delta.Delta.PartialJSON == "" {
		t.Fatalf("delta event = %+v", delta)
	}
}

func assertAnthropicStreamEvent(t *testing.T, events []RawStreamEvent, eventName string, payloadType string) {
	t.Helper()
	if len(events) != 1 || events[0].Event != eventName {
		t.Fatalf("events = %+v", events)
	}
	var decoded anthropicStreamEvent
	if err := json.Unmarshal(events[0].Data, &decoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	if decoded.Type != payloadType {
		t.Fatalf("payload type = %q, want %q", decoded.Type, payloadType)
	}
}

func TestReasoningConvertsAcrossProtocols(t *testing.T) {
	chatAdapter := NewOpenAIChatAdapter()
	chatReq, err := chatAdapter.DecodeRequest([]byte(`{
		"model":"gpt-5.4",
		"reasoning_effort":"medium",
		"messages":[{"role":"user","content":"Hello"}]
	}`))
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	if chatReq.Reasoning == nil || !*chatReq.Reasoning {
		t.Fatalf("chat reasoning = %+v", chatReq.Reasoning)
	}

	anthropicRaw, err := NewAnthropicMessagesAdapter().EncodeRequest(chatReq, EncodeRequestOptions{Model: "claude"})
	if err != nil {
		t.Fatalf("Anthropic EncodeRequest() error = %v", err)
	}
	var anthropicDecoded map[string]any
	if err := json.Unmarshal(anthropicRaw, &anthropicDecoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	thinking := anthropicDecoded["thinking"].(map[string]any)
	if thinking["type"] != "enabled" || thinking["budget_tokens"] != float64(defaultThinkingBudgetTokens) {
		t.Fatalf("anthropic thinking = %+v", thinking)
	}

	anthropicReq, err := NewAnthropicMessagesAdapter().DecodeRequest([]byte(`{
		"model":"claude",
		"max_tokens":1024,
		"thinking":{"type":"enabled","budget_tokens":1024},
		"messages":[{"role":"user","content":"Hello"}]
	}`))
	if err != nil {
		t.Fatalf("Anthropic DecodeRequest() error = %v", err)
	}
	chatRaw, err := chatAdapter.EncodeRequest(anthropicReq, EncodeRequestOptions{Model: "gpt"})
	if err != nil {
		t.Fatalf("Chat EncodeRequest() error = %v", err)
	}
	var chatDecoded map[string]any
	if err := json.Unmarshal(chatRaw, &chatDecoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	if chatDecoded["reasoning_effort"] != "medium" {
		t.Fatalf("chat reasoning_effort = %v", chatDecoded["reasoning_effort"])
	}

	responsesRaw, err := NewOpenAIResponsesAdapter().EncodeRequest(anthropicReq, EncodeRequestOptions{Model: "gpt"})
	if err != nil {
		t.Fatalf("Responses EncodeRequest() error = %v", err)
	}
	var responsesDecoded map[string]any
	if err := json.Unmarshal(responsesRaw, &responsesDecoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	reasoning := responsesDecoded["reasoning"].(map[string]any)
	if reasoning["effort"] != "medium" {
		t.Fatalf("responses reasoning = %+v", reasoning)
	}
}

func TestAnthropicMessagesEncodeImageFileIDOnlyKeepsWarningText(t *testing.T) {
	adapter := NewAnthropicMessagesAdapter()
	req := &LLMRequest{
		Model: "claude",
		Prompt: []Message{
			{Role: RoleUser, Parts: []Part{{Type: PartFile, File: &FilePart{Type: FileImage, FileID: "file_123"}}}},
		},
	}

	raw, err := adapter.EncodeRequest(req, EncodeRequestOptions{Model: "claude"})
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}

	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	messages := decoded["messages"].([]any)
	content := messages[0].(map[string]any)["content"].([]any)
	if len(content) != 1 || content[0].(map[string]any)["type"] != "text" || !strings.Contains(content[0].(map[string]any)["text"].(string), "file_123") {
		t.Fatalf("content = %+v", content)
	}
}

func TestAnthropicMessagesEncodeDocumentsUseStableSources(t *testing.T) {
	adapter := NewAnthropicMessagesAdapter()
	req := &LLMRequest{
		Model: "claude",
		Prompt: []Message{
			{Role: RoleUser, Parts: []Part{
				{Type: PartFile, File: &FilePart{Type: FileDocument, MediaType: "application/pdf", Data: "pdf-data", Filename: "doc.pdf"}},
				{Type: PartFile, File: &FilePart{Type: FileDocument, MediaType: "text/plain", Data: "plain text", Filename: "note.txt"}},
				{Type: PartFile, File: &FilePart{Type: FileDocument, FileID: "file_123", Filename: "remote.pdf"}},
				{Type: PartFile, File: &FilePart{Type: FileDocument, MediaType: "application/vnd.ms-excel", Data: "xls", Filename: "sheet.xls"}},
			}},
		},
	}

	raw, err := adapter.EncodeRequest(req, EncodeRequestOptions{Model: "claude"})
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}

	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	messages := decoded["messages"].([]any)
	content := messages[0].(map[string]any)["content"].([]any)
	if len(content) != 4 {
		t.Fatalf("content = %+v", content)
	}
	pdf := content[0].(map[string]any)
	pdfSource := pdf["source"].(map[string]any)
	if pdf["type"] != "document" || pdfSource["type"] != "base64" || pdfSource["media_type"] != "application/pdf" || pdfSource["data"] != "pdf-data" {
		t.Fatalf("pdf = %+v", pdf)
	}
	text := content[1].(map[string]any)
	textSource := text["source"].(map[string]any)
	if text["type"] != "document" || textSource["type"] != "text" || textSource["media_type"] != "text/plain" || textSource["data"] != "plain text" {
		t.Fatalf("text = %+v", text)
	}
	remoteWarning := content[2].(map[string]any)
	if remoteWarning["type"] != "text" || !strings.Contains(remoteWarning["text"].(string), "remote.pdf") {
		t.Fatalf("remote warning = %+v", remoteWarning)
	}
	xlsWarning := content[3].(map[string]any)
	if xlsWarning["type"] != "text" || !strings.Contains(xlsWarning["text"].(string), "sheet.xls") {
		t.Fatalf("xls warning = %+v", xlsWarning)
	}
}
