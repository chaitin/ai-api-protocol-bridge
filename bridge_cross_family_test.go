package protocolbridge

import (
	"encoding/json"
	"strings"
	"testing"
)

func containsString(value string, needle string) bool {
	return strings.Contains(value, needle)
}

func TestAnthropicToOpenAIResponsesBridgeEncodeUpstreamRequest(t *testing.T) {
	bridge, ok := NewCrossFamilyBridge(ProtocolAnthropicMessages, "openai")
	if !ok {
		t.Fatal("NewCrossFamilyBridge() ok = false, want true")
	}

	maxTokens := 64
	reasoning := true
	parallelToolCalls := false
	reasoningSummary := "auto"
	req := &LLMRequest{
		Protocol: ProtocolAnthropicMessages,
		Model:    "claude-sonnet",
		Prompt: []Message{
			{Role: RoleSystem, Parts: []Part{{Type: PartText, Text: &TextPart{Text: "You are helpful."}}}},
			{Role: RoleUser, Parts: []Part{{Type: PartReasoning, Reasoning: &ReasoningPart{Text: "Need concise answer.", Signature: "sig_1"}}, {Type: PartText, Text: &TextPart{Text: "Hello"}}}},
			{Role: RoleAssistant, Parts: []Part{{Type: PartReasoning, Reasoning: &ReasoningPart{Text: "Encrypted thinking", Redacted: "encrypted"}}, {Type: PartText, Text: &TextPart{Text: "Hello back"}}, {Type: PartToolCall, ToolCall: &ToolCallPart{ToolCallID: "toolu_1", ToolName: "get_weather", Input: map[string]any{"city": "Shanghai"}}}}},
			{Role: RoleTool, Parts: []Part{{Type: PartToolResult, ToolResult: &ToolResultPart{ToolCallID: "toolu_1", Output: ToolResultOutput{Type: ToolResultContent, Content: []Part{{Type: PartText, Text: &TextPart{Text: "failed"}}}}}}}},
		},
		MaxOutputTokens:   &maxTokens,
		Reasoning:         &reasoning,
		ReasoningEffort:   "xhigh",
		ReasoningSummary:  reasoningSummary,
		Tools:             []Tool{{Type: ToolFunction, Name: "no_args", InputSchema: map[string]any{"type": "object", "additionalProperties": map[string]any{}}}},
		Include:           []string{"reasoning.encrypted_content"},
		ParallelToolCalls: &parallelToolCalls,
		Metadata:          map[string]string{"user_id": "session-1"},
	}

	raw, err := bridge.EncodeUpstreamRequest(req, EncodeRequestOptions{Model: "gpt-5.4"})
	if err != nil {
		t.Fatalf("EncodeUpstreamRequest() error = %v", err)
	}

	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	if decoded["model"] != "gpt-5.4" {
		t.Fatalf("model = %v", decoded["model"])
	}
	if decoded["instructions"] != "You are helpful." {
		t.Fatalf("instructions = %v", decoded["instructions"])
	}
	inputs := decoded["input"].([]any)
	userReasoning := inputs[0].(map[string]any)
	userMessage := inputs[1].(map[string]any)
	userContent := userMessage["content"].([]any)
	if userMessage["role"] != "user" || len(userContent) != 1 || userContent[0].(map[string]any)["type"] != "input_text" {
		t.Fatalf("user input = %+v", userMessage)
	}
	if userReasoning["type"] != "reasoning" || userReasoning["encrypted_content"] != "sig_1" {
		t.Fatalf("user reasoning item = %+v", userReasoning)
	}
	reasoningItem := inputs[2].(map[string]any)
	if reasoningItem["type"] != "reasoning" || reasoningItem["encrypted_content"] != "encrypted" {
		t.Fatalf("reasoning item = %+v", reasoningItem)
	}
	assistantMessage := inputs[3].(map[string]any)
	if assistantMessage["type"] != "message" || assistantMessage["role"] != "assistant" || assistantMessage["status"] != "completed" {
		t.Fatalf("assistant message = %+v", assistantMessage)
	}
	toolCall := inputs[4].(map[string]any)
	if toolCall["type"] != "function_call" || toolCall["call_id"] != "toolu_1" {
		t.Fatalf("tool call = %+v", toolCall)
	}
	toolResult := inputs[5].(map[string]any)
	outputItems := toolResult["output"].([]any)
	if toolResult["type"] != "function_call_output" || outputItems[0].(map[string]any)["type"] != "input_text" || outputItems[0].(map[string]any)["text"] != "failed" {
		t.Fatalf("tool result = %+v", toolResult)
	}

	ordered := anthropicBridgeInputItemsForMessage(Message{Role: RoleUser, Parts: []Part{
		{Type: PartToolResult, ToolResult: &ToolResultPart{ToolCallID: "toolu_1", Output: ToolResultOutput{Type: ToolResultText, Text: "failed"}}},
		{Type: PartText, Text: &TextPart{Text: "next user text"}},
	}})
	if len(ordered) != 2 || ordered[0].Type != "function_call_output" || ordered[1].Type != "message" {
		t.Fatalf("tool result ordering = %+v", ordered)
	}
	reasoningConfig := decoded["reasoning"].(map[string]any)
	if reasoningConfig["effort"] != "xhigh" || reasoningConfig["summary"] != "auto" {
		t.Fatalf("reasoning config = %+v", reasoningConfig)
	}
	if _, ok := decoded["metadata"]; ok {
		t.Fatalf("metadata should be omitted for cross-family requests: %+v", decoded["metadata"])
	}
	tools := decoded["tools"].([]any)
	parameters := tools[0].(map[string]any)["parameters"].(map[string]any)
	if parameters["type"] != "object" || parameters["properties"] == nil {
		t.Fatalf("tool parameters = %+v", parameters)
	}
	if parameters["additionalProperties"] != false {
		t.Fatalf("tool additionalProperties = %+v", parameters["additionalProperties"])
	}
	if decoded["parallel_tool_calls"] != false {
		t.Fatalf("parallel_tool_calls = %v", decoded["parallel_tool_calls"])
	}
	include := decoded["include"].([]any)
	if len(include) != 1 || include[0] != "reasoning.encrypted_content" {
		t.Fatalf("include = %+v", include)
	}
}

func TestAnthropicToOpenAIResponsesBridgePreservesEmptyReasoningSummary(t *testing.T) {
	bridge, ok := NewCrossFamilyBridge(ProtocolAnthropicMessages, "openai")
	if !ok {
		t.Fatal("NewCrossFamilyBridge() ok = false, want true")
	}

	tests := []struct {
		name             string
		content          string
		encryptedContent string
		reasoningID      string
	}{
		{
			name:             "omitted thinking",
			content:          `[{"type":"thinking","thinking":"","signature":"sig_omitted"},{"type":"text","text":"answer"}]`,
			encryptedContent: "sig_omitted",
			reasoningID:      "rs_ba7d9f29b4a92e1ee9392d44166a85d6c870be0b642bacf3f33f877c4b1fec26",
		},
		{
			name:             "redacted thinking",
			content:          `[{"type":"redacted_thinking","data":"encrypted_redacted"},{"type":"text","text":"answer"}]`,
			encryptedContent: "encrypted_redacted",
			reasoningID:      "rs_e0b6c818077e13f6c87efe822396461444f93de7062f06fc38d00a4c5a7a0a29",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rawRequest := []byte(`{"model":"claude","max_tokens":128,"messages":[{"role":"user","content":"start"},{"role":"assistant","content":` + tt.content + `},{"role":"user","content":"continue"}]}`)
			req, err := NewAnthropicMessagesAdapter().DecodeRequest(rawRequest)
			if err != nil {
				t.Fatalf("DecodeRequest() error = %v", err)
			}

			rawUpstream, err := bridge.EncodeUpstreamRequest(req, EncodeRequestOptions{Model: "gpt-5.4"})
			if err != nil {
				t.Fatalf("EncodeUpstreamRequest() error = %v", err)
			}

			var upstream struct {
				Input []map[string]json.RawMessage `json:"input"`
			}
			if err := json.Unmarshal(rawUpstream, &upstream); err != nil {
				t.Fatalf("json.Unmarshal() error = %v", err)
			}
			if len(upstream.Input) != 4 {
				t.Fatalf("input = %s", rawUpstream)
			}

			var reasoning map[string]json.RawMessage
			for _, item := range upstream.Input {
				var itemType string
				if err := json.Unmarshal(item["type"], &itemType); err != nil {
					t.Fatalf("decode input item type: %v; item = %s", err, mustMarshalJSON(item))
				}
				if itemType == "reasoning" {
					reasoning = item
					continue
				}
				if _, exists := item["summary"]; exists {
					t.Fatalf("%s item unexpectedly contains summary: %s", itemType, mustMarshalJSON(item))
				}
			}
			if reasoning == nil {
				t.Fatalf("reasoning item not found: %s", rawUpstream)
			}
			if got := string(reasoning["summary"]); got != "[]" {
				t.Fatalf("reasoning summary = %s, want []; item = %s", got, mustMarshalJSON(reasoning))
			}
			var itemID string
			if err := json.Unmarshal(reasoning["id"], &itemID); err != nil {
				t.Fatalf("decode reasoning id: %v", err)
			}
			if itemID != tt.reasoningID {
				t.Fatalf("reasoning id = %q, want %q", itemID, tt.reasoningID)
			}
			var encryptedContent string
			if err := json.Unmarshal(reasoning["encrypted_content"], &encryptedContent); err != nil {
				t.Fatalf("decode encrypted_content: %v", err)
			}
			if encryptedContent != tt.encryptedContent {
				t.Fatalf("encrypted_content = %q, want %q", encryptedContent, tt.encryptedContent)
			}
		})
	}
}

func mustMarshalJSON(value any) string {
	raw, _ := json.Marshal(value)
	return string(raw)
}

func TestAnthropicToOpenAIResponsesBridgeDecodeUpstreamResponse(t *testing.T) {
	bridge, ok := NewCrossFamilyBridge(ProtocolAnthropicMessages, "openai")
	if !ok {
		t.Fatal("NewCrossFamilyBridge() ok = false, want true")
	}

	raw := []byte(`{
		"id":"resp_1",
		"object":"response",
		"status":"completed",
		"model":"gpt-5.4",
		"output":[
			{"type":"reasoning","summary":[{"type":"summary_text","text":"I should answer briefly."}]},
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hello back"}]},
			{"type":"function_call","call_id":"toolu_1","name":"get_weather","arguments":"{\"city\":\"Shanghai\"}","status":"completed"},
			{"type":"function_call_output","call_id":"toolu_1","output":"failed","status":"completed"}
		],
		"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15,"input_tokens_details":{"cached_tokens":3},"output_tokens_details":{"reasoning_tokens":2}}
	}`)

	resp, err := bridge.DecodeUpstreamResponse(raw)
	if err != nil {
		t.Fatalf("DecodeUpstreamResponse() error = %v", err)
	}
	if resp.Protocol != ProtocolOpenAIResponses {
		t.Fatalf("Protocol = %q", resp.Protocol)
	}
	if len(resp.Content) != 4 {
		t.Fatalf("len(Content) = %d", len(resp.Content))
	}
	if resp.Content[0].Reasoning == nil || resp.Content[0].Reasoning.Text != "I should answer briefly." {
		t.Fatalf("reasoning = %+v", resp.Content[0])
	}
	if resp.Content[3].ToolResult == nil || resp.Content[3].ToolResult.Output.Text != "failed" {
		t.Fatalf("tool result = %+v", resp.Content[3])
	}
	billingUsage := resp.BillingUsage()
	if billingUsage.InputTokens != 7 || billingUsage.CachedInputTokens != 3 || billingUsage.OutputTokens != 5 {
		t.Fatalf("billing usage = %+v", billingUsage)
	}
}

func TestAnthropicInboundOpenAIResponsesResponseEncodesEncryptedReasoningSummaryAsThinking(t *testing.T) {
	bridge, ok := NewCrossFamilyBridge(ProtocolAnthropicMessages, "openai")
	if !ok {
		t.Fatal("NewCrossFamilyBridge() ok = false, want true")
	}
	upstreamRaw := []byte(`{
		"id":"resp_1",
		"object":"response",
		"status":"completed",
		"model":"gpt-5.4",
		"output":[
			{"type":"reasoning","encrypted_content":"enc_1","summary":[{"type":"summary_text","text":"summary"}]},
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hello back"}]}
		]
	}`)

	resp, err := bridge.DecodeUpstreamResponse(upstreamRaw)
	if err != nil {
		t.Fatalf("DecodeUpstreamResponse() error = %v", err)
	}
	raw, err := NewAnthropicMessagesAdapter().EncodeResponse(resp, EncodeResponseOptions{Model: "claude-sonnet"})
	if err != nil {
		t.Fatalf("EncodeResponse() error = %v", err)
	}

	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	content := decoded["content"].([]any)
	reasoning := content[0].(map[string]any)
	if reasoning["type"] != "thinking" || reasoning["thinking"] != "summary" || reasoning["signature"] != "enc_1" {
		t.Fatalf("reasoning content = %+v", reasoning)
	}
}

func TestAnthropicToOpenAIResponsesBridgeEncodesContentFilterAsAnthropicRefusal(t *testing.T) {
	bridge, ok := NewCrossFamilyBridge(ProtocolAnthropicMessages, "openai")
	if !ok {
		t.Fatal("NewCrossFamilyBridge() ok = false, want true")
	}

	upstreamRaw := []byte(`{
		"id":"resp_1",
		"object":"response",
		"status":"incomplete",
		"incomplete_details":{"reason":"content_filter"},
		"model":"gpt-5.4",
		"output_text":"partial"
	}`)
	resp, err := bridge.DecodeUpstreamResponse(upstreamRaw)
	if err != nil {
		t.Fatalf("DecodeUpstreamResponse() error = %v", err)
	}

	raw, err := NewAnthropicMessagesAdapter().EncodeResponse(resp, EncodeResponseOptions{Model: "claude-sonnet"})
	if err != nil {
		t.Fatalf("EncodeResponse() error = %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	if decoded["stop_reason"] != "refusal" {
		t.Fatalf("response = %+v", decoded)
	}
	stopDetails := decoded["stop_details"].(map[string]any)
	if stopDetails["type"] != "refusal" {
		t.Fatalf("stop_details = %+v", stopDetails)
	}
}

func TestOpenAIResponsesToAnthropicBridgeEncodeUpstreamRequest(t *testing.T) {
	bridge, ok := NewCrossFamilyBridge(ProtocolOpenAIResponses, "anthropic")
	if !ok {
		t.Fatal("NewCrossFamilyBridge() ok = false, want true")
	}

	maxTokens := 64
	reasoning := true
	strict := true
	parallelToolCalls := false
	req := &LLMRequest{
		Protocol: ProtocolOpenAIResponses,
		Model:    "gpt-5.4",
		Prompt: []Message{
			{Role: RoleSystem, Parts: []Part{{Type: PartText, Text: &TextPart{Text: "You are helpful."}}}},
			{Role: RoleDeveloper, Parts: []Part{{Type: PartText, Text: &TextPart{Text: "Follow project rules."}}}},
			{Role: RoleUser, Parts: []Part{{Type: PartReasoning, Reasoning: &ReasoningPart{Text: "Need concise answer."}}, {Type: PartText, Text: &TextPart{Text: "Hello"}}}},
			{Role: RoleAssistant, Parts: []Part{{Type: PartText, Text: &TextPart{Text: "Hello back"}}, {Type: PartToolCall, ToolCall: &ToolCallPart{ToolCallID: "call_1", ToolName: "get_weather", Input: map[string]any{"city": "Shanghai"}}}}},
			{Role: RoleTool, Parts: []Part{{Type: PartToolResult, ToolResult: &ToolResultPart{ToolCallID: "call_1", Output: ToolResultOutput{Type: ToolResultText, Text: "Sunny"}}}}},
		},
		MaxOutputTokens:   &maxTokens,
		Reasoning:         &reasoning,
		ResponseFormat:    &ResponseFormat{Type: ResponseFormatJSON, Schema: map[string]any{"type": "object"}, Strict: &strict},
		Tools:             []Tool{{Type: ToolFunction, Name: "get_weather", Description: "Get weather.", InputSchema: map[string]any{"type": "object"}, Strict: &strict}},
		ToolChoice:        &ToolChoice{Type: ToolChoiceRequired},
		ParallelToolCalls: &parallelToolCalls,
		Metadata:          map[string]string{"trace_id": "trace_1"},
	}

	raw, err := bridge.EncodeUpstreamRequest(req, EncodeRequestOptions{Model: "claude-sonnet"})
	if err != nil {
		t.Fatalf("EncodeUpstreamRequest() error = %v", err)
	}

	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	if decoded["model"] != "claude-sonnet" {
		t.Fatalf("model = %v", decoded["model"])
	}
	system := decoded["system"].([]any)[0].(map[string]any)
	if system["text"] != "You are helpful.\nFollow project rules." {
		t.Fatalf("system = %+v", system)
	}
	if _, ok := decoded["thinking"]; ok {
		t.Fatalf("thinking should be omitted when max_tokens cannot fit a valid budget: %+v", decoded["thinking"])
	}
	toolChoice := decoded["tool_choice"].(map[string]any)
	if toolChoice["type"] != "any" || toolChoice["disable_parallel_tool_use"] != true {
		t.Fatalf("tool_choice = %+v", toolChoice)
	}
	messages := decoded["messages"].([]any)
	assistant := messages[1].(map[string]any)
	assistantContent := assistant["content"].([]any)
	if assistant["role"] != "assistant" || assistantContent[1].(map[string]any)["type"] != "tool_use" {
		t.Fatalf("assistant = %+v", assistant)
	}
	toolResult := messages[2].(map[string]any)["content"].([]any)[0].(map[string]any)
	if toolResult["type"] != "tool_result" || toolResult["content"] != "Sunny" {
		t.Fatalf("tool_result = %+v", toolResult)
	}
	outputConfig := decoded["output_config"].(map[string]any)
	if outputConfig["format"].(map[string]any)["type"] != "json_schema" {
		t.Fatalf("output_config = %+v", outputConfig)
	}
	if _, ok := decoded["metadata"]; ok {
		t.Fatalf("metadata should be omitted for cross-family requests: %+v", decoded["metadata"])
	}
}

func TestOpenAIResponsesToAnthropicBridgeMergesConsecutiveToolResults(t *testing.T) {
	bridge, ok := NewCrossFamilyBridge(ProtocolOpenAIResponses, "anthropic")
	if !ok {
		t.Fatal("NewCrossFamilyBridge() ok = false, want true")
	}

	req := &LLMRequest{
		Protocol: ProtocolOpenAIResponses,
		Model:    "gpt-5.4",
		Prompt: []Message{
			{Role: RoleUser, Parts: []Part{{Type: PartText, Text: &TextPart{Text: "Read a.txt and b.txt"}}}},
			{Role: RoleAssistant, Parts: []Part{
				{Type: PartToolCall, ToolCall: &ToolCallPart{ToolCallID: "call_00", ToolName: "read", Input: map[string]any{"path": "a.txt"}}},
				{Type: PartToolCall, ToolCall: &ToolCallPart{ToolCallID: "call_01", ToolName: "read", Input: map[string]any{"path": "b.txt"}}},
			}},
			{Role: RoleTool, Parts: []Part{{Type: PartToolResult, ToolResult: &ToolResultPart{ToolCallID: "call_00", Output: ToolResultOutput{Type: ToolResultText, Text: "aaa"}}}}},
			{Role: RoleTool, Parts: []Part{{Type: PartToolResult, ToolResult: &ToolResultPart{ToolCallID: "call_01", Output: ToolResultOutput{Type: ToolResultText, Text: "bbb"}}}}},
		},
	}

	raw, err := bridge.EncodeUpstreamRequest(req, EncodeRequestOptions{Model: "claude-sonnet"})
	if err != nil {
		t.Fatalf("EncodeUpstreamRequest() error = %v", err)
	}

	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	messages := decoded["messages"].([]any)
	if len(messages) != 3 {
		t.Fatalf("messages length = %d, messages = %+v", len(messages), messages)
	}
	toolResults := messages[2].(map[string]any)
	if toolResults["role"] != "user" {
		t.Fatalf("tool results role = %v", toolResults["role"])
	}
	content := toolResults["content"].([]any)
	if len(content) != 2 {
		t.Fatalf("tool result content = %+v", content)
	}
	first := content[0].(map[string]any)
	second := content[1].(map[string]any)
	if first["type"] != "tool_result" || first["tool_use_id"] != "call_00" || first["content"] != "aaa" {
		t.Fatalf("first tool result = %+v", first)
	}
	if second["type"] != "tool_result" || second["tool_use_id"] != "call_01" || second["content"] != "bbb" {
		t.Fatalf("second tool result = %+v", second)
	}
}

func TestOpenAIResponsesToAnthropicBridgeMergesDecodedParallelToolItems(t *testing.T) {
	req, err := NewOpenAIResponsesAdapter().DecodeRequest([]byte(`{
		"model":"gpt-5.4",
		"input":[
			{"role":"user","content":"Read a.txt and b.txt"},
			{"type":"function_call","call_id":"call_00","name":"read","arguments":"{\"path\":\"a.txt\"}","status":"completed"},
			{"type":"function_call","call_id":"call_01","name":"read","arguments":"{\"path\":\"b.txt\"}","status":"completed"},
			{"type":"function_call_output","call_id":"call_00","output":"aaa"},
			{"type":"function_call_output","call_id":"call_01","output":"bbb"}
		]
	}`))
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}

	bridge, ok := NewCrossFamilyBridge(ProtocolOpenAIResponses, "anthropic")
	if !ok {
		t.Fatal("NewCrossFamilyBridge() ok = false, want true")
	}
	raw, err := bridge.EncodeUpstreamRequest(req, EncodeRequestOptions{Model: "claude-sonnet"})
	if err != nil {
		t.Fatalf("EncodeUpstreamRequest() error = %v", err)
	}

	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	messages := decoded["messages"].([]any)
	if len(messages) != 3 {
		t.Fatalf("messages length = %d, messages = %+v", len(messages), messages)
	}
	assistantContent := messages[1].(map[string]any)["content"].([]any)
	if len(assistantContent) != 2 {
		t.Fatalf("assistant content = %+v", assistantContent)
	}
	if assistantContent[0].(map[string]any)["id"] != "call_00" || assistantContent[1].(map[string]any)["id"] != "call_01" {
		t.Fatalf("assistant content = %+v", assistantContent)
	}
	toolResultContent := messages[2].(map[string]any)["content"].([]any)
	if len(toolResultContent) != 2 {
		t.Fatalf("tool result content = %+v", toolResultContent)
	}
	if toolResultContent[0].(map[string]any)["tool_use_id"] != "call_00" || toolResultContent[1].(map[string]any)["tool_use_id"] != "call_01" {
		t.Fatalf("tool result content = %+v", toolResultContent)
	}
}

func TestOpenAIResponsesToAnthropicBridgeDecodeUpstreamResponse(t *testing.T) {
	bridge, ok := NewCrossFamilyBridge(ProtocolOpenAIResponses, "anthropic")
	if !ok {
		t.Fatal("NewCrossFamilyBridge() ok = false, want true")
	}

	raw := []byte(`{
		"id":"msg_1",
		"type":"message",
		"role":"assistant",
		"model":"claude-sonnet",
		"content":[
			{"type":"thinking","thinking":"I should answer briefly.","signature":"sig_1"},
			{"type":"text","text":"Hello back"},
			{"type":"tool_result","tool_use_id":"call_1","content":"failed","is_error":true}
		],
		"stop_reason":"end_turn",
		"usage":{"input_tokens":10,"output_tokens":5,"cache_creation_input_tokens":3,"cache_read_input_tokens":4}
	}`)

	resp, err := bridge.DecodeUpstreamResponse(raw)
	if err != nil {
		t.Fatalf("DecodeUpstreamResponse() error = %v", err)
	}
	if resp.Protocol != ProtocolOpenAIResponses {
		t.Fatalf("Protocol = %q", resp.Protocol)
	}
	if resp.Content[0].Reasoning == nil || resp.Content[0].Reasoning.Text != "I should answer briefly." {
		t.Fatalf("reasoning = %+v", resp.Content[0])
	}
	billingUsage := resp.BillingUsage()
	if billingUsage.InputTokens != 13 || billingUsage.CachedInputTokens != 4 || billingUsage.OutputTokens != 5 {
		t.Fatalf("billing usage = %+v", billingUsage)
	}
}

func TestOpenAIResponsesToAnthropicBridgeDecodesRefusalAsContentFilter(t *testing.T) {
	bridge, ok := NewCrossFamilyBridge(ProtocolOpenAIResponses, "anthropic")
	if !ok {
		t.Fatal("NewCrossFamilyBridge() ok = false, want true")
	}

	raw := []byte(`{
		"id":"msg_1",
		"type":"message",
		"role":"assistant",
		"model":"claude-sonnet",
		"content":[{"type":"text","text":"I can't help with that."}],
		"stop_reason":"refusal",
		"usage":{"input_tokens":10,"output_tokens":5}
	}`)

	resp, err := bridge.DecodeUpstreamResponse(raw)
	if err != nil {
		t.Fatalf("DecodeUpstreamResponse() error = %v", err)
	}
	if resp.FinishReason != FinishContentFilter {
		t.Fatalf("FinishReason = %q, want %q", resp.FinishReason, FinishContentFilter)
	}
}

func TestOpenAIResponsesToAnthropicBridgeOmitsUnsupportedReasoningEffortAndProviderTool(t *testing.T) {
	bridge, ok := NewCrossFamilyBridge(ProtocolOpenAIResponses, "anthropic")
	if !ok {
		t.Fatal("NewCrossFamilyBridge() ok = false, want true")
	}
	maxTokens := 4096
	reasoning := true
	req := &LLMRequest{
		Protocol:        ProtocolOpenAIResponses,
		Model:           "gpt-5.4",
		Prompt:          []Message{{Role: RoleUser, Parts: []Part{{Type: PartText, Text: &TextPart{Text: "Search"}}}}},
		MaxOutputTokens: &maxTokens,
		Reasoning:       &reasoning,
		ReasoningEffort: "xhigh",
		Tools:           []Tool{{Type: ToolProviderDefined, Name: "web_search_preview", Config: map[string]any{"max_uses": float64(2)}}},
	}

	raw, err := bridge.EncodeUpstreamRequest(req, EncodeRequestOptions{Model: "claude-sonnet"})
	if err != nil {
		t.Fatalf("EncodeUpstreamRequest() error = %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	if _, ok := decoded["output_config"]; ok {
		t.Fatalf("output_config should be omitted without a supported format: %+v", decoded["output_config"])
	}
	if _, ok := decoded["tools"]; ok {
		t.Fatalf("provider-defined tools should be omitted for Anthropic requests: %+v", decoded["tools"])
	}
	systemBlocks := decoded["system"].([]any)
	system := systemBlocks[0].(map[string]any)["text"].(string)
	if system == "" || !containsString(system, "web_search_preview") {
		t.Fatalf("system warning should mention omitted provider tool: %q", system)
	}
}

func TestOpenAIResponsesToAnthropicBridgePreservesUnsupportedFileAsWarningText(t *testing.T) {
	bridge, ok := NewCrossFamilyBridge(ProtocolOpenAIResponses, "anthropic")
	if !ok {
		t.Fatal("NewCrossFamilyBridge() ok = false, want true")
	}
	req := &LLMRequest{
		Protocol: ProtocolOpenAIResponses,
		Model:    "gpt-5.4",
		Prompt: []Message{{Role: RoleUser, Parts: []Part{{Type: PartFile, File: &FilePart{
			Type:     FileDocument,
			Filename: "report.docx",
			FileID:   "file_123",
		}}}}},
	}

	raw, err := bridge.EncodeUpstreamRequest(req, EncodeRequestOptions{Model: "claude-sonnet"})
	if err != nil {
		t.Fatalf("EncodeUpstreamRequest() error = %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	messages := decoded["messages"].([]any)
	content := messages[0].(map[string]any)["content"].([]any)
	warningBlock := content[0].(map[string]any)
	if warningBlock["type"] != "text" || !containsString(warningBlock["text"].(string), "report.docx") {
		t.Fatalf("unsupported file warning block = %+v", warningBlock)
	}
}

func TestOpenAIChatToAnthropicBridgeEncodeUpstreamRequest(t *testing.T) {
	bridge, ok := NewCrossFamilyBridge(ProtocolOpenAIChat, "anthropic")
	if !ok {
		t.Fatal("NewCrossFamilyBridge() ok = false, want true")
	}

	maxTokens := 128
	temperature := 0.7
	reasoning := true
	strict := true
	parallelToolCalls := false
	req := &LLMRequest{
		Protocol: ProtocolOpenAIChat,
		Model:    "gpt-4.1",
		Prompt: []Message{
			{Role: RoleSystem, Parts: []Part{{Type: PartText, Text: &TextPart{Text: "You are helpful."}}}},
			{Role: RoleDeveloper, Parts: []Part{{Type: PartText, Text: &TextPart{Text: "Follow project rules."}}}},
			{Role: RoleUser, Parts: []Part{{Type: PartText, Text: &TextPart{Text: "Hello"}}, {Type: PartFile, File: &FilePart{Type: FileImage, MediaType: "image/png", Data: "abc", Detail: "high"}}}},
			{Role: RoleAssistant, Parts: []Part{{Type: PartToolCall, ToolCall: &ToolCallPart{ToolCallID: "call_1", ToolName: "get_weather", Input: map[string]any{"city": "Shanghai"}}}}},
			{Role: RoleTool, Parts: []Part{{Type: PartToolResult, ToolResult: &ToolResultPart{ToolCallID: "call_1", Output: ToolResultOutput{Type: ToolResultText, Text: "Sunny"}}}}},
		},
		MaxOutputTokens:   &maxTokens,
		Temperature:       &temperature,
		StopSequences:     []string{"END"},
		Reasoning:         &reasoning,
		ResponseFormat:    &ResponseFormat{Type: ResponseFormatJSON, Schema: map[string]any{"type": "object"}, Strict: &strict},
		Tools:             []Tool{{Type: ToolFunction, Name: "get_weather", Description: "Get weather.", InputSchema: map[string]any{"type": "object"}, Strict: &strict}},
		ToolChoice:        &ToolChoice{Type: ToolChoiceRequired},
		ParallelToolCalls: &parallelToolCalls,
		Stream:            true,
		Metadata:          map[string]string{"trace_id": "trace_1"},
	}

	raw, err := bridge.EncodeUpstreamRequest(req, EncodeRequestOptions{Model: "claude-sonnet"})
	if err != nil {
		t.Fatalf("EncodeUpstreamRequest() error = %v", err)
	}

	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	if decoded["model"] != "claude-sonnet" {
		t.Fatalf("model = %v", decoded["model"])
	}
	if decoded["stream"] != true {
		t.Fatalf("stream = %v", decoded["stream"])
	}
	if decoded["max_tokens"] != float64(128) {
		t.Fatalf("max_tokens = %v", decoded["max_tokens"])
	}
	stopSequences := decoded["stop_sequences"].([]any)
	if len(stopSequences) != 1 || stopSequences[0] != "END" {
		t.Fatalf("stop_sequences = %+v", stopSequences)
	}
	system := decoded["system"].([]any)[0].(map[string]any)
	if system["text"] != "You are helpful.\nFollow project rules." {
		t.Fatalf("system = %+v", system)
	}
	toolChoice := decoded["tool_choice"].(map[string]any)
	if toolChoice["type"] != "any" || toolChoice["disable_parallel_tool_use"] != true {
		t.Fatalf("tool_choice = %+v", toolChoice)
	}
	messages := decoded["messages"].([]any)
	userContent := messages[0].(map[string]any)["content"].([]any)
	if userContent[0].(map[string]any)["type"] != "text" || userContent[1].(map[string]any)["type"] != "image" {
		t.Fatalf("user content = %+v", userContent)
	}
	toolUse := messages[1].(map[string]any)["content"].([]any)[0].(map[string]any)
	if toolUse["type"] != "tool_use" || toolUse["name"] != "get_weather" {
		t.Fatalf("tool_use = %+v", toolUse)
	}
	toolResult := messages[2].(map[string]any)["content"].([]any)[0].(map[string]any)
	if toolResult["type"] != "tool_result" || toolResult["content"] != "Sunny" {
		t.Fatalf("tool_result = %+v", toolResult)
	}
	if _, ok := decoded["thinking"]; ok {
		t.Fatalf("thinking should be omitted when max_tokens cannot fit a valid budget: %+v", decoded["thinking"])
	}
	if _, ok := decoded["metadata"]; ok {
		t.Fatalf("metadata should be omitted for cross-family requests: %+v", decoded["metadata"])
	}
}

func TestOpenAIChatToAnthropicBridgeMergesParallelToolResults(t *testing.T) {
	chatReq, err := NewOpenAIChatAdapter().DecodeRequest([]byte(`{
		"model":"deepseek-v4-pro",
		"messages":[
			{"role":"user","content":"读取 a.txt 和 b.txt"},
			{"role":"assistant","content":"","tool_calls":[
				{"id":"call_00","type":"function","function":{"name":"read","arguments":"{\"path\":\"a.txt\"}"}},
				{"id":"call_01","type":"function","function":{"name":"read","arguments":"{\"path\":\"b.txt\"}"}}
			]},
			{"role":"tool","tool_call_id":"call_00","content":"aaa"},
			{"role":"tool","tool_call_id":"call_01","content":"bbb"}
		],
		"tools":[{"type":"function","function":{"name":"read","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}}],
		"stream":true
	}`))
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}

	bridge, ok := NewCrossFamilyBridge(ProtocolOpenAIChat, "anthropic")
	if !ok {
		t.Fatal("NewCrossFamilyBridge() ok = false, want true")
	}
	raw, err := bridge.EncodeUpstreamRequest(chatReq, EncodeRequestOptions{Model: "claude-sonnet"})
	if err != nil {
		t.Fatalf("EncodeUpstreamRequest() error = %v", err)
	}

	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	messages := decoded["messages"].([]any)
	if len(messages) != 3 {
		t.Fatalf("messages length = %d, messages = %+v", len(messages), messages)
	}
	assistantContent := messages[1].(map[string]any)["content"].([]any)
	if len(assistantContent) != 2 {
		t.Fatalf("assistant content = %+v", assistantContent)
	}
	toolResults := messages[2].(map[string]any)
	if toolResults["role"] != "user" {
		t.Fatalf("tool results role = %v", toolResults["role"])
	}
	content := toolResults["content"].([]any)
	if len(content) != 2 {
		t.Fatalf("tool result content = %+v", content)
	}
	if content[0].(map[string]any)["tool_use_id"] != "call_00" || content[1].(map[string]any)["tool_use_id"] != "call_01" {
		t.Fatalf("tool result content = %+v", content)
	}
}

func TestOpenAIChatToAnthropicBridgeDisablesThinkingForUnsignedToolContinuation(t *testing.T) {
	bridge, ok := NewCrossFamilyBridge(ProtocolOpenAIChat, "anthropic")
	if !ok {
		t.Fatal("NewCrossFamilyBridge() ok = false, want true")
	}

	maxTokens := 4096
	reasoning := true
	req := &LLMRequest{
		Protocol:        ProtocolOpenAIChat,
		Model:           "deepseek-v4-pro",
		MaxOutputTokens: &maxTokens,
		Reasoning:       &reasoning,
		Prompt: []Message{
			{Role: RoleUser, Parts: []Part{{Type: PartText, Text: &TextPart{Text: "读取 a.txt"}}}},
			{Role: RoleAssistant, Parts: []Part{{Type: PartToolCall, ToolCall: &ToolCallPart{ToolCallID: "call_00", ToolName: "read", Input: map[string]any{"path": "a.txt"}}}}},
			{Role: RoleTool, Parts: []Part{{Type: PartToolResult, ToolResult: &ToolResultPart{ToolCallID: "call_00", Output: ToolResultOutput{Type: ToolResultText, Text: "aaa"}}}}},
		},
	}

	raw, err := bridge.EncodeUpstreamRequest(req, EncodeRequestOptions{Model: "claude-sonnet"})
	if err != nil {
		t.Fatalf("EncodeUpstreamRequest() error = %v", err)
	}

	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	if _, ok := decoded["thinking"]; ok {
		t.Fatalf("thinking should be omitted for unsigned OpenAI tool continuations: %+v", decoded["thinking"])
	}
	messages := decoded["messages"].([]any)
	assistantContent := messages[1].(map[string]any)["content"].([]any)
	for _, block := range assistantContent {
		if block.(map[string]any)["type"] == "thinking" {
			t.Fatalf("unsigned reasoning history should not be encoded as thinking: %+v", assistantContent)
		}
	}
}

func TestOpenAIResponsesToAnthropicBridgePreservesReasoningForToolContinuation(t *testing.T) {
	adapter := NewOpenAIResponsesAdapter()
	rawRequest := []byte(`{
		"model":"gpt-5.4",
		"max_output_tokens":4096,
		"reasoning":{"effort":"medium"},
		"input":[
			{"type":"reasoning","id":"rs_1","status":"completed","encrypted_content":"enc_1","summary":[{"type":"summary_text","text":"Need the weather tool."}]},
			{"type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"I will check the weather."}]},
			{"type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"Hangzhou\"}","status":"completed"},
			{"type":"function_call_output","call_id":"call_1","output":"Cloudy","status":"completed"}
		]
	}`)
	req, err := adapter.DecodeRequest(rawRequest)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	bridge, ok := NewCrossFamilyBridge(ProtocolOpenAIResponses, "anthropic")
	if !ok {
		t.Fatal("NewCrossFamilyBridge() ok = false, want true")
	}

	raw, err := bridge.EncodeUpstreamRequest(req, EncodeRequestOptions{Model: "claude-sonnet"})
	if err != nil {
		t.Fatalf("EncodeUpstreamRequest() error = %v", err)
	}

	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	if decoded["thinking"] == nil {
		t.Fatalf("thinking should remain enabled when signed reasoning history is present: %+v", decoded)
	}
	messages := decoded["messages"].([]any)
	assistantContent := messages[0].(map[string]any)["content"].([]any)
	if len(assistantContent) != 3 {
		t.Fatalf("assistant content = %+v", assistantContent)
	}
	reasoningBlock := assistantContent[0].(map[string]any)
	if reasoningBlock["type"] != "thinking" || reasoningBlock["thinking"] != "Need the weather tool." || reasoningBlock["signature"] != "enc_1" {
		t.Fatalf("reasoning block = %+v", reasoningBlock)
	}
	if assistantContent[1].(map[string]any)["type"] != "text" || assistantContent[2].(map[string]any)["type"] != "tool_use" {
		t.Fatalf("assistant content order = %+v", assistantContent)
	}
}

func TestOpenAIChatDecodeRequestPreservesReasoningContentInCanonicalRequest(t *testing.T) {
	req, err := NewOpenAIChatAdapter().DecodeRequest([]byte(`{
		"model":"deepseek-v4-pro",
		"messages":[
			{"role":"assistant","reasoning_content":"Need the weather tool.","content":"I will check the weather."}
		]
	}`))
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	if len(req.Prompt) != 1 || len(req.Prompt[0].Parts) != 2 {
		t.Fatalf("prompt = %+v", req.Prompt)
	}
	reasoning := req.Prompt[0].Parts[0].Reasoning
	if reasoning == nil || reasoning.Text != "Need the weather tool." {
		t.Fatalf("reasoning = %+v", reasoning)
	}
	if req.Prompt[0].Parts[1].Text == nil || req.Prompt[0].Parts[1].Text.Text != "I will check the weather." {
		t.Fatalf("text part = %+v", req.Prompt[0].Parts[1])
	}
}

func TestAnthropicInboundOpenAIResponsesUpstreamStreamBridge(t *testing.T) {
	bridge, ok := NewCrossFamilyBridge(ProtocolAnthropicMessages, "openai")
	if !ok {
		t.Fatal("NewCrossFamilyBridge() ok = false, want true")
	}
	decoder, err := bridge.NewStreamDecoder(StreamDecodeOptions{})
	if err != nil {
		t.Fatalf("NewStreamDecoder() error = %v", err)
	}
	encoder, err := bridge.NewStreamEncoder(StreamEncodeOptions{Model: "claude-sonnet"})
	if err != nil {
		t.Fatalf("NewStreamEncoder() error = %v", err)
	}

	parts, err := decoder.Decode(RawStreamEvent{Event: "response.created", Data: []byte(`{"type":"response.created","response":{"id":"resp_1","object":"response","status":"in_progress","model":"gpt-5.4","usage":{"input_tokens":10,"input_tokens_details":{"cached_tokens":3},"output_tokens":0}}}`)})
	if err != nil {
		t.Fatalf("Decode(response.created) error = %v", err)
	}
	startEvents, err := encoder.Encode(parts[0])
	if err != nil {
		t.Fatalf("Encode(start) error = %v", err)
	}
	var start anthropicStreamEvent
	if err := json.Unmarshal(startEvents[0].Data, &start); err != nil {
		t.Fatalf("json.Unmarshal(start) error = %v", err)
	}
	if start.Type != "message_start" || start.Message == nil || start.Message.Usage.InputTokens == nil || *start.Message.Usage.InputTokens != 7 {
		t.Fatalf("start = %+v", start)
	}
	if start.Message.Usage.CacheReadInputTokens == nil || *start.Message.Usage.CacheReadInputTokens != 3 {
		t.Fatalf("start cache read = %+v", start.Message.Usage)
	}

	parts, err = decoder.Decode(RawStreamEvent{Event: "response.output_item.added", Data: []byte(`{"type":"response.output_item.added","item":{"id":"rs_1","type":"reasoning","status":"in_progress"}}`)})
	if err != nil {
		t.Fatalf("Decode(reasoning start) error = %v", err)
	}
	reasonStartEvents, err := encoder.Encode(parts[0])
	if err != nil {
		t.Fatalf("Encode(reasoning start) error = %v", err)
	}
	var reasonStart anthropicStreamEvent
	if err := json.Unmarshal(reasonStartEvents[0].Data, &reasonStart); err != nil {
		t.Fatalf("json.Unmarshal(reasonStart) error = %v", err)
	}
	if reasonStart.ContentBlock == nil || reasonStart.ContentBlock.Type != "thinking" {
		t.Fatalf("reasonStart = %+v", reasonStart)
	}

	parts, err = decoder.Decode(RawStreamEvent{Event: "response.reasoning_summary_text.delta", Data: []byte(`{"type":"response.reasoning_summary_text.delta","delta":"think"}`)})
	if err != nil {
		t.Fatalf("Decode(reasoning delta) error = %v", err)
	}
	reasonDeltaEvents, err := encoder.Encode(parts[0])
	if err != nil {
		t.Fatalf("Encode(reasoning delta) error = %v", err)
	}
	var reasonDelta anthropicStreamEvent
	if err := json.Unmarshal(reasonDeltaEvents[0].Data, &reasonDelta); err != nil {
		t.Fatalf("json.Unmarshal(reasonDelta) error = %v", err)
	}
	if reasonDelta.Delta == nil || reasonDelta.Delta.Type != "thinking_delta" || reasonDelta.Delta.Thinking != "think" {
		t.Fatalf("reasonDelta = %+v", reasonDelta)
	}

	parts, err = decoder.Decode(RawStreamEvent{Event: "response.output_item.added", Data: []byte(`{"type":"response.output_item.added","item":{"id":"fc_1","type":"function_call","status":"in_progress","name":"get_weather","call_id":"call_1"}}`)})
	if err != nil {
		t.Fatalf("Decode(tool start) error = %v", err)
	}
	toolStartEvents, err := encoder.Encode(parts[0])
	if err != nil {
		t.Fatalf("Encode(tool start) error = %v", err)
	}
	var toolStart anthropicStreamEvent
	if err := json.Unmarshal(toolStartEvents[0].Data, &toolStart); err != nil {
		t.Fatalf("json.Unmarshal(toolStart) error = %v", err)
	}
	if toolStart.ContentBlock == nil || toolStart.ContentBlock.Type != "tool_use" || toolStart.ContentBlock.Name != "get_weather" {
		t.Fatalf("toolStart = %+v", toolStart)
	}

	parts, err = decoder.Decode(RawStreamEvent{Event: "response.function_call_arguments.delta", Data: []byte(`{"type":"response.function_call_arguments.delta","delta":"{\"city\":\"Shanghai\"}"}`)})
	if err != nil {
		t.Fatalf("Decode(tool delta) error = %v", err)
	}
	toolDeltaEvents, err := encoder.Encode(parts[0])
	if err != nil {
		t.Fatalf("Encode(tool delta) error = %v", err)
	}
	var toolDelta anthropicStreamEvent
	if err := json.Unmarshal(toolDeltaEvents[0].Data, &toolDelta); err != nil {
		t.Fatalf("json.Unmarshal(toolDelta) error = %v", err)
	}
	if toolDelta.Delta == nil || toolDelta.Delta.Type != "input_json_delta" || toolDelta.Delta.PartialJSON != `{"city":"Shanghai"}` {
		t.Fatalf("toolDelta = %+v", toolDelta)
	}

	parts, err = decoder.Decode(RawStreamEvent{Event: "response.completed", Data: []byte(`{"type":"response.completed","response":{"id":"resp_1","status":"completed","usage":{"input_tokens":10,"input_tokens_details":{"cached_tokens":3},"output_tokens":5}}}`)})
	if err != nil {
		t.Fatalf("Decode(response.completed) error = %v", err)
	}
	finishEvents, err := encoder.Encode(parts[0])
	if err != nil {
		t.Fatalf("Encode(finish) error = %v", err)
	}
	var finish anthropicStreamEvent
	if err := json.Unmarshal(finishEvents[0].Data, &finish); err != nil {
		t.Fatalf("json.Unmarshal(finish) error = %v", err)
	}
	if finish.Type != "message_delta" || finish.Usage == nil || finish.Usage.InputTokens == nil || *finish.Usage.InputTokens != 7 {
		t.Fatalf("finish = %+v", finish)
	}
	if finish.Usage.CacheReadInputTokens == nil || *finish.Usage.CacheReadInputTokens != 3 || finish.Usage.OutputTokens == nil || *finish.Usage.OutputTokens != 5 {
		t.Fatalf("finish usage = %+v", finish.Usage)
	}
}

func TestAnthropicInboundOpenAIResponsesStreamDoesNotDuplicateContentStart(t *testing.T) {
	bridge, ok := NewCrossFamilyBridge(ProtocolAnthropicMessages, "openai")
	if !ok {
		t.Fatal("NewCrossFamilyBridge() ok = false, want true")
	}
	decoder, err := bridge.NewStreamDecoder(StreamDecodeOptions{})
	if err != nil {
		t.Fatalf("NewStreamDecoder() error = %v", err)
	}
	encoder, err := bridge.NewStreamEncoder(StreamEncodeOptions{Model: "claude-sonnet"})
	if err != nil {
		t.Fatalf("NewStreamEncoder() error = %v", err)
	}

	parts, err := decoder.Decode(RawStreamEvent{Event: "response.created", Data: []byte(`{"type":"response.created","response":{"id":"resp_1","status":"in_progress","model":"gpt-5.4"}}`)})
	if err != nil {
		t.Fatalf("Decode(response.created) error = %v", err)
	}
	for _, part := range parts {
		if _, err := encoder.Encode(part); err != nil {
			t.Fatalf("Encode(response.created) error = %v", err)
		}
	}

	parts, err = decoder.Decode(RawStreamEvent{Event: "response.output_item.added", Data: []byte(`{"type":"response.output_item.added","item":{"id":"msg_1","type":"message","role":"assistant","status":"in_progress"}}`)})
	if err != nil {
		t.Fatalf("Decode(output_item.added) error = %v", err)
	}
	var startEvents []RawStreamEvent
	for _, part := range parts {
		encoded, err := encoder.Encode(part)
		if err != nil {
			t.Fatalf("Encode(output_item.added) error = %v", err)
		}
		startEvents = append(startEvents, encoded...)
	}
	if len(startEvents) != 1 || startEvents[0].Event != "content_block_start" {
		t.Fatalf("output_item.added events = %+v", startEvents)
	}

	parts, err = decoder.Decode(RawStreamEvent{Event: "response.content_part.added", Data: []byte(`{"type":"response.content_part.added","item_id":"msg_1","content_part":{"type":"output_text","text":""}}`)})
	if err != nil {
		t.Fatalf("Decode(content_part.added) error = %v", err)
	}
	var duplicateEvents []RawStreamEvent
	for _, part := range parts {
		encoded, err := encoder.Encode(part)
		if err != nil {
			t.Fatalf("Encode(content_part.added) error = %v", err)
		}
		duplicateEvents = append(duplicateEvents, encoded...)
	}
	if len(duplicateEvents) != 0 {
		t.Fatalf("content_part.added should not emit duplicate Anthropic start, events = %+v", duplicateEvents)
	}
}

func TestAnthropicInboundOpenAIResponsesStreamPreservesReasoningSignature(t *testing.T) {
	bridge, ok := NewCrossFamilyBridge(ProtocolAnthropicMessages, "openai")
	if !ok {
		t.Fatal("NewCrossFamilyBridge() ok = false, want true")
	}
	decoder, err := bridge.NewStreamDecoder(StreamDecodeOptions{})
	if err != nil {
		t.Fatalf("NewStreamDecoder() error = %v", err)
	}
	encoder, err := bridge.NewStreamEncoder(StreamEncodeOptions{Model: "claude-sonnet"})
	if err != nil {
		t.Fatalf("NewStreamEncoder() error = %v", err)
	}

	parts, err := decoder.Decode(RawStreamEvent{Event: "response.output_item.added", Data: []byte(`{"type":"response.output_item.added","item":{"id":"rs_1","type":"reasoning","status":"in_progress"}}`)})
	if err != nil {
		t.Fatalf("Decode(reasoning added) error = %v", err)
	}
	for _, part := range parts {
		if _, err := encoder.Encode(part); err != nil {
			t.Fatalf("Encode(reasoning added) error = %v", err)
		}
	}

	parts, err = decoder.Decode(RawStreamEvent{Event: "response.reasoning_summary_text.done", Data: []byte(`{"type":"response.reasoning_summary_text.done","item_id":"rs_1"}`)})
	if err != nil {
		t.Fatalf("Decode(reasoning summary done) error = %v", err)
	}
	var summaryDoneEvents []RawStreamEvent
	for _, part := range parts {
		encoded, err := encoder.Encode(part)
		if err != nil {
			t.Fatalf("Encode(reasoning summary done) error = %v", err)
		}
		summaryDoneEvents = append(summaryDoneEvents, encoded...)
	}
	if len(summaryDoneEvents) != 0 {
		t.Fatalf("reasoning summary done should wait for output_item.done, events = %+v", summaryDoneEvents)
	}

	parts, err = decoder.Decode(RawStreamEvent{Event: "response.reasoning.done", Data: []byte(`{"type":"response.reasoning.done","item_id":"rs_1"}`)})
	if err != nil {
		t.Fatalf("Decode(reasoning done event) error = %v", err)
	}
	var reasoningDoneEvents []RawStreamEvent
	for _, part := range parts {
		encoded, err := encoder.Encode(part)
		if err != nil {
			t.Fatalf("Encode(reasoning done event) error = %v", err)
		}
		reasoningDoneEvents = append(reasoningDoneEvents, encoded...)
	}
	if len(reasoningDoneEvents) != 0 {
		t.Fatalf("reasoning done should wait for output_item.done, events = %+v", reasoningDoneEvents)
	}

	parts, err = decoder.Decode(RawStreamEvent{Event: "response.output_item.done", Data: []byte(`{"type":"response.output_item.done","item":{"id":"rs_1","type":"reasoning","status":"completed","encrypted_content":"enc_1"}}`)})
	if err != nil {
		t.Fatalf("Decode(reasoning done) error = %v", err)
	}
	var events []RawStreamEvent
	for _, part := range parts {
		encoded, err := encoder.Encode(part)
		if err != nil {
			t.Fatalf("Encode(reasoning done) error = %v", err)
		}
		events = append(events, encoded...)
	}
	if len(events) != 2 {
		t.Fatalf("reasoning done events = %+v", events)
	}
	var signature anthropicStreamEvent
	if err := json.Unmarshal(events[0].Data, &signature); err != nil {
		t.Fatalf("json.Unmarshal(signature) error = %v", err)
	}
	if signature.Type != "content_block_delta" || signature.Delta == nil || signature.Delta.Type != "signature_delta" || signature.Delta.Signature != "enc_1" {
		t.Fatalf("signature event = %+v", signature)
	}
	if events[1].Event != "content_block_stop" {
		t.Fatalf("reasoning stop event = %+v", events[1])
	}
}

func TestAnthropicInboundOpenAIResponsesStreamRefusal(t *testing.T) {
	bridge, ok := NewCrossFamilyBridge(ProtocolAnthropicMessages, "openai")
	if !ok {
		t.Fatal("NewCrossFamilyBridge() ok = false, want true")
	}
	decoder, err := bridge.NewStreamDecoder(StreamDecodeOptions{})
	if err != nil {
		t.Fatalf("NewStreamDecoder() error = %v", err)
	}
	encoder, err := bridge.NewStreamEncoder(StreamEncodeOptions{Model: "claude-sonnet"})
	if err != nil {
		t.Fatalf("NewStreamEncoder() error = %v", err)
	}

	events := make([]RawStreamEvent, 0)
	for _, raw := range []RawStreamEvent{
		{Event: "response.created", Data: []byte(`{"type":"response.created","response":{"id":"resp_1","status":"in_progress","model":"gpt-5.4"}}`)},
		{Event: "response.output_item.added", Data: []byte(`{"type":"response.output_item.added","item":{"id":"msg_1","type":"message","role":"assistant","status":"in_progress"}}`)},
		{Event: "response.content_part.added", Data: []byte(`{"type":"response.content_part.added","item_id":"msg_1","content_part":{"type":"refusal","text":""}}`)},
		{Event: "response.refusal.delta", Data: []byte(`{"type":"response.refusal.delta","item_id":"msg_1","delta":"I can't help."}`)},
		{Event: "response.refusal.done", Data: []byte(`{"type":"response.refusal.done","item_id":"msg_1"}`)},
		{Event: "response.completed", Data: []byte(`{"type":"response.completed","response":{"id":"resp_1","status":"completed","usage":{"input_tokens":10,"output_tokens":5}}}`)},
	} {
		parts, err := decoder.Decode(raw)
		if err != nil {
			t.Fatalf("Decode(%s) error = %v", raw.Event, err)
		}
		for _, part := range parts {
			encoded, err := encoder.Encode(part)
			if err != nil {
				t.Fatalf("Encode(%s) error = %v", raw.Event, err)
			}
			events = append(events, encoded...)
		}
	}

	var sawDelta bool
	var finish anthropicStreamEvent
	for _, event := range events {
		var decoded anthropicStreamEvent
		if err := json.Unmarshal(event.Data, &decoded); err != nil {
			t.Fatalf("json.Unmarshal(%s) error = %v", event.Event, err)
		}
		if decoded.Type == "content_block_delta" && decoded.Delta != nil && decoded.Delta.Text == "I can't help." {
			sawDelta = true
		}
		if decoded.Type == "message_delta" {
			finish = decoded
		}
	}
	if !sawDelta {
		t.Fatalf("refusal text delta not found, events = %+v", events)
	}
	if finish.Delta == nil || finish.Delta.StopReason != "refusal" {
		t.Fatalf("finish = %+v", finish)
	}
}

func TestOpenAIResponsesInboundAnthropicUpstreamStreamBridge(t *testing.T) {
	bridge, ok := NewCrossFamilyBridge(ProtocolOpenAIResponses, "anthropic")
	if !ok {
		t.Fatal("NewCrossFamilyBridge() ok = false, want true")
	}
	decoder, err := bridge.NewStreamDecoder(StreamDecodeOptions{})
	if err != nil {
		t.Fatalf("NewStreamDecoder() error = %v", err)
	}
	encoder, err := bridge.NewStreamEncoder(StreamEncodeOptions{Model: "gpt-5.4"})
	if err != nil {
		t.Fatalf("NewStreamEncoder() error = %v", err)
	}

	parts, err := decoder.Decode(RawStreamEvent{Event: "message_start", Data: []byte(`{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet","content":[],"usage":{"input_tokens":10,"cache_creation_input_tokens":3,"cache_read_input_tokens":4,"output_tokens":0}}}`)})
	if err != nil {
		t.Fatalf("Decode(start) error = %v", err)
	}
	startEvents, err := encoder.Encode(parts[0])
	if err != nil {
		t.Fatalf("Encode(start) error = %v", err)
	}
	var start openAIResponsesStreamEvent
	if err := json.Unmarshal(startEvents[0].Data, &start); err != nil {
		t.Fatalf("json.Unmarshal(start) error = %v", err)
	}
	if start.Type != "response.created" || start.Response == nil || start.Response.Usage == nil || start.Response.Usage.InputTokens == nil || *start.Response.Usage.InputTokens != 17 {
		t.Fatalf("start = %+v", start)
	}
	if start.Response.Usage.InputTokensDetails == nil || start.Response.Usage.InputTokensDetails.CachedTokens == nil || *start.Response.Usage.InputTokensDetails.CachedTokens != 4 {
		t.Fatalf("start usage = %+v", start.Response.Usage)
	}

	parts, err = decoder.Decode(RawStreamEvent{Event: "content_block_start", Data: []byte(`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`)})
	if err != nil {
		t.Fatalf("Decode(reasoning start) error = %v", err)
	}
	if _, err := encoder.Encode(parts[0]); err != nil {
		t.Fatalf("Encode(reasoning start) error = %v", err)
	}
	parts, err = decoder.Decode(RawStreamEvent{Event: "content_block_delta", Data: []byte(`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"think"}}`)})
	if err != nil {
		t.Fatalf("Decode(reasoning delta) error = %v", err)
	}
	reasonEvents, err := encoder.Encode(parts[0])
	if err != nil {
		t.Fatalf("Encode(reasoning delta) error = %v", err)
	}
	var reason openAIResponsesStreamEvent
	if err := json.Unmarshal(reasonEvents[0].Data, &reason); err != nil {
		t.Fatalf("json.Unmarshal(reason) error = %v", err)
	}
	if reason.Type != "response.reasoning_summary_text.delta" || reason.Delta != "think" {
		t.Fatalf("reason = %+v", reason)
	}

	parts, err = decoder.Decode(RawStreamEvent{Event: "content_block_start", Data: []byte(`{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"tool_1","name":"Write","input":{}}}`)})
	if err != nil {
		t.Fatalf("Decode(tool start) error = %v", err)
	}
	toolStartEvents, err := encoder.Encode(parts[0])
	if err != nil {
		t.Fatalf("Encode(tool start) error = %v", err)
	}
	var toolStart openAIResponsesStreamEvent
	if err := json.Unmarshal(toolStartEvents[0].Data, &toolStart); err != nil {
		t.Fatalf("json.Unmarshal(toolStart) error = %v", err)
	}
	if toolStart.Item == nil || toolStart.Item.Type != "function_call" || toolStart.Item.CallID != "tool_1" {
		t.Fatalf("toolStart = %+v", toolStart)
	}

	parts, err = decoder.Decode(RawStreamEvent{Event: "content_block_delta", Data: []byte(`{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"file_path\":\"/tmp/hello.c\"}"}}`)})
	if err != nil {
		t.Fatalf("Decode(tool delta) error = %v", err)
	}
	toolDeltaEvents, err := encoder.Encode(parts[0])
	if err != nil {
		t.Fatalf("Encode(tool delta) error = %v", err)
	}
	var toolDelta openAIResponsesStreamEvent
	if err := json.Unmarshal(toolDeltaEvents[0].Data, &toolDelta); err != nil {
		t.Fatalf("json.Unmarshal(toolDelta) error = %v", err)
	}
	if toolDelta.Type != "response.function_call_arguments.delta" || toolDelta.Delta != `{"file_path":"/tmp/hello.c"}` {
		t.Fatalf("toolDelta = %+v", toolDelta)
	}

	parts, err = decoder.Decode(RawStreamEvent{Event: "message_delta", Data: []byte(`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}`)})
	if err != nil {
		t.Fatalf("Decode(message_delta) error = %v", err)
	}
	finishEvents, err := encoder.Encode(parts[0])
	if err != nil {
		t.Fatalf("Encode(finish) error = %v", err)
	}
	var finish openAIResponsesStreamEvent
	if err := json.Unmarshal(finishEvents[0].Data, &finish); err != nil {
		t.Fatalf("json.Unmarshal(finish) error = %v", err)
	}
	if finish.Type != "response.completed" || finish.Response == nil || finish.Response.Usage == nil || finish.Response.Usage.InputTokens == nil || *finish.Response.Usage.InputTokens != 17 {
		t.Fatalf("finish = %+v", finish)
	}
	if finish.Response.ID != "msg_1" {
		t.Fatalf("finish response id = %q, want msg_1", finish.Response.ID)
	}
	if finish.Response.Usage.OutputTokens == nil || *finish.Response.Usage.OutputTokens != 5 {
		t.Fatalf("finish usage = %+v", finish.Response.Usage)
	}
}

func TestOpenAIChatInboundAnthropicUpstreamStreamBridge(t *testing.T) {
	bridge, ok := NewCrossFamilyBridge(ProtocolOpenAIChat, "anthropic")
	if !ok {
		t.Fatal("NewCrossFamilyBridge() ok = false, want true")
	}
	decoder, err := bridge.NewStreamDecoder(StreamDecodeOptions{})
	if err != nil {
		t.Fatalf("NewStreamDecoder() error = %v", err)
	}
	encoder, err := bridge.NewStreamEncoder(StreamEncodeOptions{Model: "gpt-4.1"})
	if err != nil {
		t.Fatalf("NewStreamEncoder() error = %v", err)
	}

	parts, err := decoder.Decode(RawStreamEvent{Event: "message_start", Data: []byte(`{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet","content":[],"usage":{"input_tokens":10,"cache_creation_input_tokens":3,"cache_read_input_tokens":4,"output_tokens":0}}}`)})
	if err != nil {
		t.Fatalf("Decode(start) error = %v", err)
	}
	startEvents, err := encoder.Encode(parts[0])
	if err != nil {
		t.Fatalf("Encode(start) error = %v", err)
	}
	var start openAIChatStreamChunk
	if err := json.Unmarshal(startEvents[0].Data, &start); err != nil {
		t.Fatalf("json.Unmarshal(start) error = %v", err)
	}
	if start.Usage != nil {
		t.Fatalf("start usage = %+v, want nil", start.Usage)
	}

	parts, err = decoder.Decode(RawStreamEvent{Event: "content_block_delta", Data: []byte(`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"think"}}`)})
	if err != nil {
		t.Fatalf("Decode(reasoning delta) error = %v", err)
	}
	reasonEvents, err := encoder.Encode(parts[0])
	if err != nil {
		t.Fatalf("Encode(reasoning delta) error = %v", err)
	}
	var reason openAIChatStreamChunk
	if err := json.Unmarshal(reasonEvents[0].Data, &reason); err != nil {
		t.Fatalf("json.Unmarshal(reason) error = %v", err)
	}
	if len(reason.Choices) != 1 || reason.Choices[0].Delta == nil || reason.Choices[0].Delta.Reasoning == nil || *reason.Choices[0].Delta.Reasoning != "think" {
		t.Fatalf("reason = %+v", reason)
	}

	parts, err = decoder.Decode(RawStreamEvent{Event: "content_block_start", Data: []byte(`{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"tool_1","name":"Write","input":{}}}`)})
	if err != nil {
		t.Fatalf("Decode(tool start) error = %v", err)
	}
	toolStartEvents, err := encoder.Encode(parts[0])
	if err != nil {
		t.Fatalf("Encode(tool start) error = %v", err)
	}
	var toolStart openAIChatStreamChunk
	if err := json.Unmarshal(toolStartEvents[0].Data, &toolStart); err != nil {
		t.Fatalf("json.Unmarshal(toolStart) error = %v", err)
	}
	toolCalls := toolStart.Choices[0].Delta.ToolCalls
	if len(toolCalls) != 1 || toolCalls[0].ID != "tool_1" || toolCalls[0].Function.Name != "Write" {
		t.Fatalf("toolStart = %+v", toolStart)
	}

	parts, err = decoder.Decode(RawStreamEvent{Event: "content_block_delta", Data: []byte(`{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"file_path\":\"/tmp/hello.c\"}"}}`)})
	if err != nil {
		t.Fatalf("Decode(tool delta) error = %v", err)
	}
	toolDeltaEvents, err := encoder.Encode(parts[0])
	if err != nil {
		t.Fatalf("Encode(tool delta) error = %v", err)
	}
	var toolDelta openAIChatStreamChunk
	if err := json.Unmarshal(toolDeltaEvents[0].Data, &toolDelta); err != nil {
		t.Fatalf("json.Unmarshal(toolDelta) error = %v", err)
	}
	toolCalls = toolDelta.Choices[0].Delta.ToolCalls
	if len(toolCalls) != 1 || toolCalls[0].Function.Arguments != `{"file_path":"/tmp/hello.c"}` {
		t.Fatalf("toolDelta = %+v", toolDelta)
	}

	parts, err = decoder.Decode(RawStreamEvent{Event: "message_delta", Data: []byte(`{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}`)})
	if err != nil {
		t.Fatalf("Decode(message_delta) error = %v", err)
	}
	finishEvents, err := encoder.Encode(parts[0])
	if err != nil {
		t.Fatalf("Encode(finish) error = %v", err)
	}
	var finish openAIChatStreamChunk
	if err := json.Unmarshal(finishEvents[0].Data, &finish); err != nil {
		t.Fatalf("json.Unmarshal(finish) error = %v", err)
	}
	if len(finish.Choices) != 1 || finish.Choices[0].FinishReason == nil || *finish.Choices[0].FinishReason != "tool_calls" {
		t.Fatalf("finish = %+v", finish)
	}
	if finish.Usage != nil {
		t.Fatalf("finish usage = %+v, want nil", finish.Usage)
	}
	closeEvents, err := encoder.Close()
	if err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if len(closeEvents) != 2 || string(closeEvents[1].Data) != "[DONE]" {
		t.Fatalf("Close events = %+v", closeEvents)
	}
	var summary openAIChatStreamChunk
	if err := json.Unmarshal(closeEvents[0].Data, &summary); err != nil {
		t.Fatalf("json.Unmarshal(summary) error = %v", err)
	}
	if len(summary.Choices) != 0 || summary.Usage == nil || summary.Usage.PromptTokens == nil || *summary.Usage.PromptTokens != 17 || summary.Usage.CompletionTokens == nil || *summary.Usage.CompletionTokens != 5 {
		t.Fatalf("summary usage = %+v", summary)
	}
	if summary.Usage.PromptTokensDetails == nil || summary.Usage.PromptTokensDetails.CachedTokens == nil || *summary.Usage.PromptTokensDetails.CachedTokens != 4 {
		t.Fatalf("summary cached tokens = %+v", summary.Usage)
	}
}
