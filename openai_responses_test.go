package protocolbridge

import (
	"encoding/json"
	"testing"
)

func TestOpenAIResponsesDecodeRequestAcceptsObjectArguments(t *testing.T) {
	adapter := NewOpenAIResponsesAdapter()
	raw := []byte(`{
		"model":"gpt-5.5",
		"input":[
			{"role":"user","type":"message","content":[{"type":"input_text","text":"hi"}]},
			{"type":"tool_search_call","call_id":"call_search","arguments":{"limit":8,"query":"spawn subagent"},"status":"completed"},
			{"type":"function_call","call_id":"call_1","name":"search","arguments":{"query":"Shanghai"},"status":"completed"}
		]
	}`)

	req, err := adapter.DecodeRequest(raw)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	if len(req.Prompt) != 2 {
		t.Fatalf("Prompt length = %d", len(req.Prompt))
	}
	toolCall := req.Prompt[1].Parts[0].ToolCall
	input, ok := toolCall.Input.(map[string]any)
	if !ok || input["query"] != "Shanghai" {
		t.Fatalf("tool call input = %#v", toolCall.Input)
	}

	var request openAIResponsesRequest
	if err := json.Unmarshal(raw, &request); err != nil {
		t.Fatalf("Unmarshal request error = %v", err)
	}
	encoded, err := json.Marshal(request.Input)
	if err != nil {
		t.Fatalf("Marshal input error = %v", err)
	}
	var items []map[string]any
	if err := json.Unmarshal(encoded, &items); err != nil {
		t.Fatalf("Unmarshal encoded input error = %v", err)
	}
	arguments, ok := items[1]["arguments"].(map[string]any)
	if !ok || arguments["query"] != "spawn subagent" || arguments["limit"] != float64(8) {
		t.Fatalf("tool_search arguments = %#v", items[1]["arguments"])
	}
}

func TestOpenAIResponsesDecodeRequest(t *testing.T) {
	adapter := NewOpenAIResponsesAdapter()
	raw := []byte(`{
		"model":"gpt-5.4",
		"instructions":"You are helpful.",
		"input":[
			{"role":"user","content":[{"type":"input_text","text":"Hello"},{"type":"input_image","image_url":"data:image/png;base64,abc","detail":"high"},{"type":"input_file","file_data":"Zm9v","filename":"notes.txt","detail":"low"}]},
			{"type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"Shanghai\"}","status":"completed"},
			{"type":"function_call_output","call_id":"call_1","output":"Sunny","status":"completed"}
		],
		"tools":[{"type":"function","name":"get_weather","description":"Get weather.","parameters":{"type":"object"},"strict":true},{"type":"web_search_preview","search_context_size":"low"}],
		"tool_choice":{"type":"function","name":"get_weather"},
		"metadata":{"trace_id":"trace_1"},
		"user":"dev-user",
		"service_tier":"flex",
		"parallel_tool_calls":false,
		"store":true,
		"previous_response_id":"resp_prev",
		"truncation":"auto",
		"include":["reasoning.encrypted_content"],
		"text":{"format":{"type":"json_schema","name":"weather","schema":{"type":"object"},"strict":true}},
		"max_output_tokens":64,
		"temperature":0.2
	}`)

	req, err := adapter.DecodeRequest(raw)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	if req.Model != "gpt-5.4" {
		t.Fatalf("Model = %q", req.Model)
	}
	if len(req.Prompt) != 4 {
		t.Fatalf("Prompt length = %d", len(req.Prompt))
	}
	if req.Prompt[0].Role != RoleSystem || req.Prompt[0].Parts[0].Text.Text != "You are helpful." {
		t.Fatalf("system message = %+v", req.Prompt[0])
	}
	if req.Prompt[1].Role != RoleUser || req.Prompt[1].Parts[0].Text.Text != "Hello" {
		t.Fatalf("user message = %+v", req.Prompt[1])
	}
	image := req.Prompt[1].Parts[1].File
	if image.Type != FileImage || image.MediaType != "image/png" || image.Data != "abc" || image.Detail != "high" {
		t.Fatalf("user image = %+v", image)
	}
	file := req.Prompt[1].Parts[2].File
	if file.Type != FileDocument || file.Data != "Zm9v" || file.Filename != "notes.txt" || file.Detail != "low" {
		t.Fatalf("user file = %+v", file)
	}
	if req.MaxOutputTokens == nil || *req.MaxOutputTokens != 64 {
		t.Fatalf("MaxOutputTokens = %v", req.MaxOutputTokens)
	}
	if req.ResponseFormat == nil || req.ResponseFormat.Type != ResponseFormatJSON || req.ResponseFormat.Schema == nil || req.ResponseFormat.Strict == nil || !*req.ResponseFormat.Strict {
		t.Fatalf("ResponseFormat = %+v", req.ResponseFormat)
	}
	toolCall := req.Prompt[2].Parts[0].ToolCall
	if toolCall.ToolCallID != "call_1" || toolCall.ToolName != "get_weather" {
		t.Fatalf("tool call = %+v", toolCall)
	}
	toolResult := req.Prompt[3].Parts[0].ToolResult
	if toolResult.ToolCallID != "call_1" || toolResult.Output.Text != "Sunny" {
		t.Fatalf("tool result = %+v", toolResult)
	}
	if len(req.Tools) != 2 || req.Tools[0].Strict == nil || !*req.Tools[0].Strict || req.Tools[1].Type != ToolProviderDefined || req.Tools[1].Name != "web_search_preview" || req.Tools[1].Config["search_context_size"] != "low" {
		t.Fatalf("tools = %+v", req.Tools)
	}
	if req.State == nil || req.State.PreviousResponseID != "resp_prev" {
		t.Fatalf("State = %+v", req.State)
	}
	if len(req.Include) != 1 || req.Include[0] != "reasoning.encrypted_content" {
		t.Fatalf("Include = %+v", req.Include)
	}
	if req.ToolChoice == nil || req.ToolChoice.Type != ToolChoiceTool || req.ToolChoice.ToolName != "get_weather" {
		t.Fatalf("tool choice = %+v", req.ToolChoice)
	}
	if req.ParallelToolCalls == nil || *req.ParallelToolCalls {
		t.Fatalf("ParallelToolCalls = %+v", req.ParallelToolCalls)
	}
	if req.ProviderOptions != nil {
		t.Fatalf("ProviderOptions = %+v", req.ProviderOptions)
	}
}

func TestOpenAIResponsesEncodeRequest(t *testing.T) {
	adapter := NewOpenAIResponsesAdapter()
	maxTokens := 64
	strict := true
	parallelToolCalls := false
	req := &LLMRequest{
		Model: "gpt-5.4",
		Prompt: []Message{
			{Role: RoleSystem, Parts: []Part{{Type: PartText, Text: &TextPart{Text: "You are helpful."}}}},
			{Role: RoleDeveloper, Parts: []Part{{Type: PartText, Text: &TextPart{Text: "Follow project rules."}}}},
			{Role: RoleSystem, Parts: []Part{{Type: PartText, Text: &TextPart{Text: "Answer in Chinese."}}}},
			{Role: RoleUser, Parts: []Part{{Type: PartText, Text: &TextPart{Text: "Hello"}}, {Type: PartFile, File: &FilePart{Type: FileImage, URL: "https://example.com/cat.png"}}, {Type: PartFile, File: &FilePart{Type: FileDocument, Data: "Zm9v", Filename: "notes.txt", Detail: "low"}}}},
			{Role: RoleAssistant, Parts: []Part{{Type: PartText, Text: &TextPart{Text: "Hello back"}}, {Type: PartToolCall, ToolCall: &ToolCallPart{ToolCallID: "call_1", ToolName: "get_weather", Input: map[string]any{"city": "Shanghai"}}}}},
			{Role: RoleTool, Parts: []Part{{Type: PartToolResult, ToolResult: &ToolResultPart{ToolCallID: "call_1", Output: ToolResultOutput{Type: ToolResultContent, Content: []Part{{Type: PartText, Text: &TextPart{Text: "Sunny"}}, {Type: PartText, Text: &TextPart{Text: "Windy"}}}}}}}},
			{Role: RoleUser, Parts: []Part{{Type: PartText, Text: &TextPart{Text: "Next question"}}}},
		},
		MaxOutputTokens:   &maxTokens,
		ResponseFormat:    &ResponseFormat{Type: ResponseFormatJSON, Name: "weather_response", Description: "Weather response JSON.", Schema: map[string]any{"type": "object", "properties": map[string]any{"weather": map[string]any{"type": "string"}}}, Strict: &strict},
		State:             &RequestState{PreviousResponseID: "resp_prev"},
		Include:           []string{"reasoning.encrypted_content"},
		Tools:             []Tool{{Type: ToolFunction, Name: "get_weather", Description: "Get weather.", InputSchema: map[string]any{"type": "object"}, Strict: &strict}, {Type: ToolProviderDefined, Name: "web_search_preview", Config: map[string]any{"search_context_size": "low"}}},
		ToolChoice:        &ToolChoice{Type: ToolChoiceTool, ToolName: "get_weather"},
		ParallelToolCalls: &parallelToolCalls,
		Metadata:          map[string]string{"trace_id": "trace_1"},
	}

	raw, err := adapter.EncodeRequest(req, EncodeRequestOptions{Model: "upstream-gpt"})
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}

	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	if decoded["model"] != "upstream-gpt" || decoded["instructions"] != "You are helpful.\nFollow project rules.\nAnswer in Chinese." {
		t.Fatalf("request = %+v", decoded)
	}
	if decoded["max_output_tokens"] != float64(64) {
		t.Fatalf("max_output_tokens = %v", decoded["max_output_tokens"])
	}
	if decoded["parallel_tool_calls"] != false {
		t.Fatalf("parallel_tool_calls = %v", decoded["parallel_tool_calls"])
	}
	for _, key := range []string{"metadata", "user", "service_tier", "store", "truncation"} {
		if _, ok := decoded[key]; ok {
			t.Fatalf("provider-specific field %q should be omitted: %+v", key, decoded)
		}
	}
	if decoded["previous_response_id"] != "resp_prev" {
		t.Fatalf("previous_response_id = %v", decoded["previous_response_id"])
	}
	include := decoded["include"].([]any)
	if len(include) != 1 || include[0] != "reasoning.encrypted_content" {
		t.Fatalf("include = %+v", include)
	}
	text := decoded["text"].(map[string]any)
	format := text["format"].(map[string]any)
	if format["type"] != "json_schema" || format["name"] != "weather_response" || format["description"] != "Weather response JSON." || format["strict"] != true {
		t.Fatalf("text.format = %+v", format)
	}
	schema := format["schema"].(map[string]any)
	if schema["type"] != "object" {
		t.Fatalf("schema = %+v", schema)
	}
	toolChoice := decoded["tool_choice"].(map[string]any)
	if toolChoice["type"] != "function" || toolChoice["name"] != "get_weather" {
		t.Fatalf("tool_choice = %+v", toolChoice)
	}
	tools := decoded["tools"].([]any)
	tool := tools[0].(map[string]any)
	if tool["strict"] != true {
		t.Fatalf("tool = %+v", tool)
	}
	providerTool := tools[1].(map[string]any)
	if providerTool["type"] != "web_search_preview" || providerTool["search_context_size"] != "low" {
		t.Fatalf("provider tool = %+v", providerTool)
	}
	inputs := decoded["input"].([]any)
	firstInput := inputs[0].(map[string]any)
	if firstInput["role"] != "user" {
		t.Fatalf("first input = %+v", firstInput)
	}
	firstContent := firstInput["content"].([]any)
	encodedImage := firstContent[1].(map[string]any)
	if encodedImage["type"] != "input_image" || encodedImage["image_url"] != "https://example.com/cat.png" || encodedImage["detail"] != "auto" {
		t.Fatalf("encoded image = %+v", encodedImage)
	}
	encodedFile := firstContent[2].(map[string]any)
	if encodedFile["type"] != "input_file" || encodedFile["file_data"] != "Zm9v" || encodedFile["filename"] != "notes.txt" || encodedFile["detail"] != "low" {
		t.Fatalf("encoded file = %+v", encodedFile)
	}
	assistantInput := inputs[1].(map[string]any)
	assistantContent := assistantInput["content"].([]any)[0].(map[string]any)
	if assistantInput["role"] != "assistant" || assistantContent["type"] != "output_text" {
		t.Fatalf("assistant input = %+v", assistantInput)
	}
	toolCall := inputs[2].(map[string]any)
	if toolCall["type"] != "function_call" || toolCall["call_id"] != "call_1" || toolCall["name"] != "get_weather" || toolCall["arguments"] != `{"city":"Shanghai"}` {
		t.Fatalf("tool call = %+v", toolCall)
	}
	toolResult := inputs[3].(map[string]any)
	outputItems := toolResult["output"].([]any)
	if toolResult["type"] != "function_call_output" || toolResult["call_id"] != "call_1" || len(outputItems) != 2 || outputItems[0].(map[string]any)["text"] != "Sunny" || outputItems[1].(map[string]any)["text"] != "Windy" {
		t.Fatalf("tool result = %+v", toolResult)
	}
	lastInput := inputs[4].(map[string]any)
	lastContent := lastInput["content"].([]any)[0].(map[string]any)
	if lastInput["role"] != "user" || lastContent["type"] != "input_text" {
		t.Fatalf("last input = %+v", lastInput)
	}
}

func TestOpenAIResponsesDecodeIncompleteResponse(t *testing.T) {
	adapter := NewOpenAIResponsesAdapter()
	raw := []byte(`{"id":"resp_1","object":"response","status":"incomplete","model":"gpt-5.4","output_text":"partial"}`)

	resp, err := adapter.DecodeResponse(raw)
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	if resp.FinishReason != FinishLength {
		t.Fatalf("FinishReason = %q", resp.FinishReason)
	}
}

func TestOpenAIResponsesEncodeResponseUsesFirstChoice(t *testing.T) {
	adapter := NewOpenAIResponsesAdapter()
	resp := &LLMResponse{
		Protocol: ProtocolOpenAIChat,
		ID:       "chatcmpl_1",
		Model:    "gpt-4.1",
		Content:  nil,
		Choices: []LLMChoice{
			{Index: 0, Role: RoleAssistant, Content: []Part{{Type: PartText, Text: &TextPart{Text: "First"}}}, FinishReason: FinishStop},
			{Index: 1, Role: RoleAssistant, Content: []Part{{Type: PartText, Text: &TextPart{Text: "Second"}}}, FinishReason: FinishLength},
		},
		FinishReason: FinishUnknown,
	}

	raw, err := adapter.EncodeResponse(resp, EncodeResponseOptions{Model: "gpt-5.4"})
	if err != nil {
		t.Fatalf("EncodeResponse() error = %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	if decoded["output_text"] != "First" || decoded["status"] != "completed" {
		t.Fatalf("response = %+v", decoded)
	}
}

func TestOpenAIResponsesEncodeRequestSkipsImageWithoutSource(t *testing.T) {
	adapter := NewOpenAIResponsesAdapter()
	req := &LLMRequest{
		Model: "gpt-5.4",
		Prompt: []Message{
			{Role: RoleUser, Parts: []Part{{Type: PartText, Text: &TextPart{Text: "Hello"}}, {Type: PartFile, File: &FilePart{Type: FileImage}}}},
		},
	}

	raw, err := adapter.EncodeRequest(req, EncodeRequestOptions{Model: "upstream-gpt"})
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}

	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	inputs := decoded["input"].([]any)
	content := inputs[0].(map[string]any)["content"].([]any)
	if len(content) != 1 || content[0].(map[string]any)["type"] != "input_text" {
		t.Fatalf("content = %+v", content)
	}
}

func TestOpenAIResponsesEncodeRequestRejectsStopSequences(t *testing.T) {
	adapter := NewOpenAIResponsesAdapter()
	req := &LLMRequest{
		Model:         "gpt-5.4",
		Prompt:        []Message{{Role: RoleUser, Parts: []Part{{Type: PartText, Text: &TextPart{Text: "Hello"}}}}},
		StopSequences: []string{"END"},
	}

	if _, err := adapter.EncodeRequest(req, EncodeRequestOptions{Model: "upstream-gpt"}); err == nil {
		t.Fatal("EncodeRequest() error = nil")
	}
}

func TestOpenAIResponsesEncodeRequestMapsReasoningBudgetToEffort(t *testing.T) {
	adapter := NewOpenAIResponsesAdapter()
	for _, tc := range []struct {
		name   string
		budget int
		effort string
	}{
		{name: "medium", budget: 1024, effort: "medium"},
		{name: "high", budget: 4096, effort: "high"},
		{name: "xhigh", budget: 8192, effort: "xhigh"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := &LLMRequest{
				Model:                 "gpt-5.4",
				Prompt:                []Message{{Role: RoleUser, Parts: []Part{{Type: PartText, Text: &TextPart{Text: "Hello"}}}}},
				ReasoningBudgetTokens: &tc.budget,
			}

			raw, err := adapter.EncodeRequest(req, EncodeRequestOptions{Model: "upstream-gpt"})
			if err != nil {
				t.Fatalf("EncodeRequest() error = %v", err)
			}

			var decoded map[string]any
			if err := json.Unmarshal(raw, &decoded); err != nil {
				t.Fatalf("Unmarshal() error = %v", err)
			}
			reasoning := decoded["reasoning"].(map[string]any)
			if reasoning["effort"] != tc.effort {
				t.Fatalf("reasoning = %+v", reasoning)
			}
			if _, ok := reasoning["max_tokens"]; ok {
				t.Fatalf("reasoning.max_tokens should be omitted: %+v", reasoning)
			}
		})
	}
}

func TestOpenAIResponsesEncodeRequestOmitsReasoningForNoneEffort(t *testing.T) {
	adapter := NewOpenAIResponsesAdapter()
	reasoning := true
	req := &LLMRequest{
		Model:           "gpt-5.4",
		Prompt:          []Message{{Role: RoleUser, Parts: []Part{{Type: PartText, Text: &TextPart{Text: "Hello"}}}}},
		Reasoning:       &reasoning,
		ReasoningEffort: "none",
	}

	raw, err := adapter.EncodeRequest(req, EncodeRequestOptions{Model: "upstream-gpt"})
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}

	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	if _, ok := decoded["reasoning"]; ok {
		t.Fatalf("reasoning should be omitted: %+v", decoded["reasoning"])
	}
}

func TestOpenAIResponsesDecodeJSONSchemaTextFormat(t *testing.T) {
	adapter := NewOpenAIResponsesAdapter()
	raw := []byte(`{
		"model":"gpt-5.4",
		"input":"Return weather as JSON.",
		"text":{"format":{"type":"json_schema","name":"weather_response","description":"Weather response JSON.","schema":{"type":"object","properties":{"weather":{"type":"string"}}}}}
	}`)

	req, err := adapter.DecodeRequest(raw)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	if req.ResponseFormat == nil || req.ResponseFormat.Type != ResponseFormatJSON || req.ResponseFormat.Name != "weather_response" || req.ResponseFormat.Description != "Weather response JSON." {
		t.Fatalf("ResponseFormat = %+v", req.ResponseFormat)
	}
	if req.ResponseFormat.Schema["type"] != "object" {
		t.Fatalf("schema = %+v", req.ResponseFormat.Schema)
	}
}

func TestOpenAIChatResponseFormatEncodesToOpenAIResponsesTextFormat(t *testing.T) {
	chatAdapter := NewOpenAIChatAdapter()
	responsesAdapter := NewOpenAIResponsesAdapter()
	chatRaw := []byte(`{
		"model":"gpt-4.1",
		"messages":[{"role":"user","content":"Return weather as JSON."}],
		"response_format":{"type":"json_schema","json_schema":{"name":"weather_response","description":"Weather response JSON.","schema":{"type":"object","properties":{"weather":{"type":"string"}}}}}
	}`)

	req, err := chatAdapter.DecodeRequest(chatRaw)
	if err != nil {
		t.Fatalf("Chat DecodeRequest() error = %v", err)
	}
	responsesRaw, err := responsesAdapter.EncodeRequest(req, EncodeRequestOptions{Model: "gpt-5.4"})
	if err != nil {
		t.Fatalf("Responses EncodeRequest() error = %v", err)
	}

	var decoded map[string]any
	if err := json.Unmarshal(responsesRaw, &decoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	text := decoded["text"].(map[string]any)
	format := text["format"].(map[string]any)
	if format["type"] != "json_schema" || format["name"] != "weather_response" || format["description"] != "Weather response JSON." {
		t.Fatalf("text.format = %+v", format)
	}
}

func TestOpenAIChatStopSequencesRejectOpenAIResponsesEncode(t *testing.T) {
	chatAdapter := NewOpenAIChatAdapter()
	responsesAdapter := NewOpenAIResponsesAdapter()
	chatRaw := []byte(`{
		"model":"gpt-4.1",
		"messages":[{"role":"user","content":"Hello"}],
		"stop":["END"]
	}`)

	req, err := chatAdapter.DecodeRequest(chatRaw)
	if err != nil {
		t.Fatalf("Chat DecodeRequest() error = %v", err)
	}
	if _, err := responsesAdapter.EncodeRequest(req, EncodeRequestOptions{Model: "gpt-5.4"}); err == nil {
		t.Fatal("Responses EncodeRequest() error = nil")
	}
}

func TestOpenAIResponsesDecodeResponse(t *testing.T) {
	adapter := NewOpenAIResponsesAdapter()
	raw := []byte(`{
		"id":"resp_1",
		"object":"response",
		"status":"completed",
		"model":"gpt-5.4",
		"output":[{"type":"reasoning","summary":[{"type":"summary_text","text":"I should answer briefly."}]},{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hello back"}]},{"type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"Shanghai\"}","status":"completed"},{"type":"function_call_output","call_id":"call_1","output":"Sunny","status":"completed"}],
		"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15,"input_tokens_details":{"cache_write_tokens":2,"cached_tokens":3},"output_tokens_details":{"reasoning_tokens":2}}
	}`)

	resp, err := adapter.DecodeResponse(raw)
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	if resp.ID != "resp_1" || resp.Model != "gpt-5.4" {
		t.Fatalf("response = %+v", resp)
	}
	if resp.Content[0].Reasoning.Text != "I should answer briefly." || resp.Content[1].Text.Text != "Hello back" || resp.Content[2].ToolCall.ToolName != "get_weather" || resp.Content[3].ToolResult.Output.Text != "Sunny" {
		t.Fatalf("content = %+v", resp.Content)
	}
	if resp.FinishReason != FinishToolCalls {
		t.Fatalf("FinishReason = %q", resp.FinishReason)
	}
	if *resp.Usage.InputTokens != 10 || *resp.Usage.OutputTokens != 5 || *resp.Usage.CachedInputTokens != 3 || *resp.Usage.CacheCreationInputTokens != 2 || *resp.Usage.ReasoningTokens != 2 {
		t.Fatalf("usage = %+v", resp.Usage)
	}
	billingUsage := resp.BillingUsage()
	if billingUsage.InputTokens != 7 || billingUsage.CachedInputTokens != 3 || billingUsage.OutputTokens != 5 {
		t.Fatalf("billing usage = %+v", billingUsage)
	}
}

func TestOpenAIResponsesDecodeResponseMergesImageGenerationUsage(t *testing.T) {
	adapter := NewOpenAIResponsesAdapter()
	raw := []byte(`{
		"id":"resp_1",
		"object":"response",
		"status":"completed",
		"model":"gpt-5.4",
		"output":[{
			"type":"image_generation_call",
			"result":"abc",
			"output_format":"png",
			"usage":{
				"input_tokens":7,
				"output_tokens":11,
				"total_tokens":18,
				"input_tokens_details":{"text_tokens":2,"image_tokens":5}
			}
		}],
		"usage":{"input_tokens":10,"output_tokens":3,"total_tokens":13,"input_tokens_details":{"cached_tokens":4}}
	}`)

	resp, err := adapter.DecodeResponse(raw)
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	billingUsage := resp.BillingUsage()
	if billingUsage.InputTokens != 13 || billingUsage.CachedInputTokens != 4 || billingUsage.OutputTokens != 14 {
		t.Fatalf("billing usage = %+v", billingUsage)
	}
	if *resp.Usage.InputTokens != 17 || *resp.Usage.OutputTokens != 14 || *resp.Usage.CachedInputTokens != 4 {
		t.Fatalf("usage = %+v", resp.Usage)
	}
}

func TestOpenAIResponsesDecodeResponseToolCallDoesNotOverrideIncompleteStatus(t *testing.T) {
	adapter := NewOpenAIResponsesAdapter()
	raw := []byte(`{
		"id":"resp_1",
		"object":"response",
		"status":"incomplete",
		"incomplete_details":{"reason":"max_output_tokens"},
		"model":"gpt-5.4",
		"output":[{"type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{}","status":"completed"}]
	}`)

	resp, err := adapter.DecodeResponse(raw)
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	if resp.FinishReason != FinishLength {
		t.Fatalf("FinishReason = %q", resp.FinishReason)
	}
}

func TestOpenAIResponsesDecodeIncompleteContentFilter(t *testing.T) {
	adapter := NewOpenAIResponsesAdapter()
	raw := []byte(`{
		"id":"resp_1",
		"object":"response",
		"status":"incomplete",
		"incomplete_details":{"reason":"content_filter"},
		"model":"gpt-5.4",
		"output_text":"partial"
	}`)

	resp, err := adapter.DecodeResponse(raw)
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	if resp.FinishReason != FinishContentFilter {
		t.Fatalf("FinishReason = %q", resp.FinishReason)
	}
}

func TestOpenAIResponsesDecodeResponseRefusal(t *testing.T) {
	adapter := NewOpenAIResponsesAdapter()
	raw := []byte(`{
		"id":"resp_1",
		"object":"response",
		"status":"completed",
		"model":"gpt-5.4",
		"output":[{"type":"message","role":"assistant","content":[{"type":"refusal","refusal":"I'm sorry, I cannot assist with that request."}]}]
	}`)

	resp, err := adapter.DecodeResponse(raw)
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	if len(resp.Content) != 1 || resp.Content[0].Type != PartRefusal || resp.Content[0].Refusal.Text != "I'm sorry, I cannot assist with that request." {
		t.Fatalf("content = %+v", resp.Content)
	}
	if resp.FinishReason != FinishContentFilter {
		t.Fatalf("FinishReason = %q", resp.FinishReason)
	}
}

func TestOpenAIResponsesRefusalEncodesToOpenAIChatRefusal(t *testing.T) {
	responsesAdapter := NewOpenAIResponsesAdapter()
	chatAdapter := NewOpenAIChatAdapter()
	responsesRaw := []byte(`{
		"id":"resp_1",
		"object":"response",
		"status":"completed",
		"model":"gpt-5.4",
		"output":[{"type":"message","role":"assistant","content":[{"type":"refusal","refusal":"I'm sorry, I cannot assist with that request."}]}]
	}`)

	resp, err := responsesAdapter.DecodeResponse(responsesRaw)
	if err != nil {
		t.Fatalf("Responses DecodeResponse() error = %v", err)
	}
	chatRaw, err := chatAdapter.EncodeResponse(resp, EncodeResponseOptions{Model: "gpt-4.1"})
	if err != nil {
		t.Fatalf("Chat EncodeResponse() error = %v", err)
	}

	var decoded map[string]any
	if err := json.Unmarshal(chatRaw, &decoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	message := decoded["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)
	if message["refusal"] != "I'm sorry, I cannot assist with that request." {
		t.Fatalf("message = %+v", message)
	}
}

func TestOpenAIResponsesToolCallEncodesToOpenAIChatToolCallsFinishReason(t *testing.T) {
	responsesAdapter := NewOpenAIResponsesAdapter()
	chatAdapter := NewOpenAIChatAdapter()
	responsesRaw := []byte(`{
		"id":"resp_1",
		"object":"response",
		"status":"completed",
		"model":"gpt-5.4",
		"output":[{"type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"Shanghai\"}","status":"completed"}]
	}`)

	resp, err := responsesAdapter.DecodeResponse(responsesRaw)
	if err != nil {
		t.Fatalf("Responses DecodeResponse() error = %v", err)
	}
	chatRaw, err := chatAdapter.EncodeResponse(resp, EncodeResponseOptions{Model: "gpt-5.4"})
	if err != nil {
		t.Fatalf("Chat EncodeResponse() error = %v", err)
	}

	var decoded map[string]any
	if err := json.Unmarshal(chatRaw, &decoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	choice := decoded["choices"].([]any)[0].(map[string]any)
	if choice["finish_reason"] != "tool_calls" {
		t.Fatalf("finish_reason = %v", choice["finish_reason"])
	}
	message := choice["message"].(map[string]any)
	toolCalls := message["tool_calls"].([]any)
	if len(toolCalls) != 1 {
		t.Fatalf("tool_calls = %+v", toolCalls)
	}
}

func TestOpenAIResponsesEncodeResponseRefusal(t *testing.T) {
	adapter := NewOpenAIResponsesAdapter()
	resp := &LLMResponse{
		Protocol:     ProtocolOpenAIResponses,
		ID:           "resp_1",
		Model:        "gpt-5.4",
		Role:         RoleAssistant,
		Content:      []Part{{Type: PartRefusal, Refusal: &RefusalPart{Text: "I'm sorry, I cannot assist with that request."}}},
		FinishReason: FinishStop,
	}

	raw, err := adapter.EncodeResponse(resp, EncodeResponseOptions{})
	if err != nil {
		t.Fatalf("EncodeResponse() error = %v", err)
	}

	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	content := decoded["output"].([]any)[0].(map[string]any)["content"].([]any)[0].(map[string]any)
	if content["type"] != "refusal" || content["refusal"] != "I'm sorry, I cannot assist with that request." {
		t.Fatalf("content = %+v", content)
	}
}

func TestOpenAIResponsesEncodeResponse(t *testing.T) {
	adapter := NewOpenAIResponsesAdapter()
	inputTokens := 10
	outputTokens := 5
	cachedInputTokens := 3
	cacheWriteTokens := 0
	totalInputTokens := inputTokens + cachedInputTokens
	reasoningTokens := 2
	resp := &LLMResponse{
		Protocol: ProtocolOpenAIResponses,
		ID:       "resp_1",
		Model:    "gpt-5.4",
		Role:     RoleAssistant,
		Content: []Part{
			{Type: PartReasoning, Reasoning: &ReasoningPart{Text: "I should answer briefly."}},
			{Type: PartText, Text: &TextPart{Text: "Hello back"}},
			{Type: PartToolCall, ToolCall: &ToolCallPart{ToolCallID: "call_1", ToolName: "get_weather", Input: map[string]any{"city": "Shanghai"}}},
			{Type: PartToolResult, ToolResult: &ToolResultPart{ToolCallID: "call_1", Output: ToolResultOutput{Type: ToolResultContent, Content: []Part{{Type: PartText, Text: &TextPart{Text: "Sunny"}}, {Type: PartText, Text: &TextPart{Text: "Windy"}}}}}},
		},
		FinishReason: FinishStop,
		Usage:        Usage{InputTokens: &totalInputTokens, OutputTokens: &outputTokens, CachedInputTokens: &cachedInputTokens, CacheCreationInputTokens: &cacheWriteTokens, ReasoningTokens: &reasoningTokens},
	}

	raw, err := adapter.EncodeResponse(resp, EncodeResponseOptions{Model: "gpt-5.4"})
	if err != nil {
		t.Fatalf("EncodeResponse() error = %v", err)
	}

	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	if decoded["object"] != "response" || decoded["status"] != "completed" || decoded["output_text"] != "Hello back" {
		t.Fatalf("response = %+v", decoded)
	}
	output := decoded["output"].([]any)
	reasoning := output[0].(map[string]any)
	if reasoning["id"] != "rs_resp_1" || reasoning["type"] != "reasoning" || reasoning["status"] != "completed" {
		t.Fatalf("reasoning = %+v", reasoning)
	}
	summary := reasoning["summary"].([]any)[0].(map[string]any)
	if summary["type"] != "summary_text" || summary["text"] != "I should answer briefly." {
		t.Fatalf("reasoning summary = %+v", summary)
	}
	message := output[1].(map[string]any)
	if message["id"] != "msg_resp_1" || message["type"] != "message" || message["role"] != "assistant" || message["status"] != "completed" {
		t.Fatalf("message = %+v", message)
	}
	messageContent := message["content"].([]any)
	if len(messageContent) != 1 {
		t.Fatalf("message content = %+v", messageContent)
	}
	outputText := messageContent[0].(map[string]any)
	if outputText["type"] != "output_text" || outputText["text"] != "Hello back" {
		t.Fatalf("output text = %+v", outputText)
	}
	toolCall := output[2].(map[string]any)
	if toolCall["type"] != "function_call" || toolCall["call_id"] != "call_1" || toolCall["name"] != "get_weather" || toolCall["arguments"] != `{"city":"Shanghai"}` {
		t.Fatalf("tool call = %+v", toolCall)
	}
	toolResult := output[3].(map[string]any)
	outputItems := toolResult["output"].([]any)
	if toolResult["type"] != "function_call_output" || toolResult["call_id"] != "call_1" || len(outputItems) != 2 || outputItems[0].(map[string]any)["text"] != "Sunny" || outputItems[1].(map[string]any)["text"] != "Windy" {
		t.Fatalf("tool result = %+v", toolResult)
	}
	usage := decoded["usage"].(map[string]any)
	if usage["input_tokens"] != float64(13) || usage["output_tokens"] != float64(5) || usage["total_tokens"] != float64(18) {
		t.Fatalf("usage = %+v", usage)
	}
	inputDetails := usage["input_tokens_details"].(map[string]any)
	if inputDetails["cached_tokens"] != float64(3) || inputDetails["cache_write_tokens"] != float64(0) {
		t.Fatalf("input_tokens_details = %+v", inputDetails)
	}
	outputDetails := usage["output_tokens_details"].(map[string]any)
	if outputDetails["reasoning_tokens"] != float64(2) {
		t.Fatalf("output_tokens_details = %+v", outputDetails)
	}
}

func TestOpenAIResponsesEncodeLengthAsIncomplete(t *testing.T) {
	adapter := NewOpenAIResponsesAdapter()
	resp := &LLMResponse{
		Protocol:     ProtocolOpenAIResponses,
		ID:           "resp_1",
		Model:        "gpt-5.4",
		Role:         RoleAssistant,
		Content:      []Part{{Type: PartText, Text: &TextPart{Text: "partial"}}},
		FinishReason: FinishLength,
	}

	raw, err := adapter.EncodeResponse(resp, EncodeResponseOptions{Model: "gpt-5.4"})
	if err != nil {
		t.Fatalf("EncodeResponse() error = %v", err)
	}

	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	if decoded["status"] != "incomplete" {
		t.Fatalf("status = %v", decoded["status"])
	}
	output := decoded["output"].([]any)
	message := output[0].(map[string]any)
	if message["status"] != "incomplete" {
		t.Fatalf("message status = %v", message["status"])
	}
}

func TestOpenAIResponsesDecodeEncryptedReasoningAndCustomTool(t *testing.T) {
	adapter := NewOpenAIResponsesAdapter()
	raw := []byte(`{
		"id":"resp_1",
		"object":"response",
		"status":"completed",
		"model":"gpt-5.4",
		"output":[
			{"type":"reasoning","encrypted_content":"enc_1","summary":[{"type":"summary_text","text":"summary"}]},
			{"type":"custom_tool_call","call_id":"call_custom","name":"grammar_tool","input":"raw input","status":"completed"}
		]
	}`)

	resp, err := adapter.DecodeResponse(raw)
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	if len(resp.Content) != 2 {
		t.Fatalf("content = %+v", resp.Content)
	}
	reasoning := resp.Content[0].Reasoning
	if reasoning == nil || reasoning.Text != "summary" || reasoning.Redacted != "" || reasoning.Signature != "enc_1" {
		t.Fatalf("reasoning = %+v", reasoning)
	}
	toolCall := resp.Content[1].ToolCall
	if toolCall == nil || toolCall.ToolCallID != "call_custom" || toolCall.ToolName != "grammar_tool" || toolCall.Input != "raw input" || !toolCall.ProviderExecuted {
		t.Fatalf("tool call = %+v", toolCall)
	}
}

func TestOpenAIResponsesDecodeTopLevelOutputTextAndImages(t *testing.T) {
	adapter := NewOpenAIResponsesAdapter()
	raw := []byte(`{
		"id":"resp_1",
		"object":"response",
		"status":"completed",
		"model":"gpt-5.4",
		"output":[
			{"type":"output_text","text":"top-level text"},
			{"type":"input_image","image_url":"https://example.com/out.png"},
			{"type":"image_generation_call","result":"abc","output_format":"jpeg"}
		]
	}`)

	resp, err := adapter.DecodeResponse(raw)
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	if len(resp.Content) != 3 {
		t.Fatalf("content = %+v", resp.Content)
	}
	if resp.Content[0].Text == nil || resp.Content[0].Text.Text != "top-level text" {
		t.Fatalf("text = %+v", resp.Content[0])
	}
	if resp.Content[1].File == nil || resp.Content[1].File.URL != "https://example.com/out.png" {
		t.Fatalf("input image = %+v", resp.Content[1])
	}
	if resp.Content[2].File == nil || resp.Content[2].File.MediaType != "image/jpeg" || resp.Content[2].File.Data != "abc" {
		t.Fatalf("generated image = %+v", resp.Content[2])
	}
}

func TestOpenAIResponsesEncodeToolResultImageAndEncryptedReasoning(t *testing.T) {
	adapter := NewOpenAIResponsesAdapter()
	resp := &LLMResponse{
		ID:    "resp_1",
		Model: "gpt-5.4",
		Role:  RoleAssistant,
		Usage: Usage{},
		Content: []Part{
			{Type: PartReasoning, Reasoning: &ReasoningPart{Text: "summary", Redacted: "enc_1"}},
			{Type: PartToolResult, ToolResult: &ToolResultPart{ToolCallID: "call_1", Output: ToolResultOutput{Type: ToolResultContent, Content: []Part{{Type: PartFile, File: &FilePart{Type: FileImage, MediaType: "image/png", Data: "abc", Detail: "high"}}}}}},
		},
		FinishReason: FinishStop,
	}

	raw, err := adapter.EncodeResponse(resp, EncodeResponseOptions{})
	if err != nil {
		t.Fatalf("EncodeResponse() error = %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	output := decoded["output"].([]any)
	reasoning := output[0].(map[string]any)
	if reasoning["encrypted_content"] != "enc_1" {
		t.Fatalf("reasoning = %+v", reasoning)
	}
	items := output[1].(map[string]any)["output"].([]any)
	image := items[0].(map[string]any)
	if image["type"] != "input_image" || image["image_url"] != "data:image/png;base64,abc" || image["detail"] != "high" {
		t.Fatalf("image output = %+v", image)
	}
}

func TestOpenAIResponsesStreamDecoder(t *testing.T) {
	decoder, err := NewOpenAIResponsesAdapter().NewStreamDecoder(StreamDecodeOptions{})
	if err != nil {
		t.Fatalf("NewStreamDecoder() error = %v", err)
	}

	parts, err := decoder.Decode(RawStreamEvent{Event: "response.created", Data: []byte(`{"type":"response.created","response":{"id":"resp_1","object":"response","status":"in_progress","model":"gpt-4"}}`)})
	if err != nil {
		t.Fatalf("Decode(created) error = %v", err)
	}
	if len(parts) != 1 || parts[0].Type != StreamStart || parts[0].ID != "resp_1" {
		t.Fatalf("created parts = %+v", parts)
	}

	parts, err = decoder.Decode(RawStreamEvent{Event: "response.output_item.added", Data: []byte(`{"type":"response.output_item.added","item":{"id":"msg_1","type":"message","role":"assistant","status":"in_progress"}}`)})
	if err != nil {
		t.Fatalf("Decode(msg added) error = %v", err)
	}
	if len(parts) != 1 || parts[0].Type != StreamTextStart {
		t.Fatalf("msg added parts = %+v", parts)
	}

	parts, err = decoder.Decode(RawStreamEvent{Event: "response.output_text.delta", Data: []byte(`{"type":"response.output_text.delta","delta":"Hi"}`)})
	if err != nil {
		t.Fatalf("Decode(text delta) error = %v", err)
	}
	if len(parts) != 1 || parts[0].Type != StreamTextDelta || parts[0].Delta != "Hi" {
		t.Fatalf("text delta parts = %+v", parts)
	}

	parts, err = decoder.Decode(RawStreamEvent{Event: "response.output_text.done", Data: []byte(`{"type":"response.output_text.done"}`)})
	if err != nil {
		t.Fatalf("Decode(text done) error = %v", err)
	}
	if len(parts) != 1 || parts[0].Type != StreamTextEnd {
		t.Fatalf("text done parts = %+v", parts)
	}

	parts, err = decoder.Decode(RawStreamEvent{Event: "response.output_item.added", Data: []byte(`{"type":"response.output_item.added","item":{"id":"fc_1","type":"function_call","name":"get_weather","call_id":"call_1","status":"in_progress"}}`)})
	if err != nil {
		t.Fatalf("Decode(tool start) error = %v", err)
	}
	if len(parts) != 1 || parts[0].Type != StreamToolInputStart || parts[0].ID != "fc_1" || parts[0].ToolCallID != "call_1" || parts[0].ToolName != "get_weather" {
		t.Fatalf("tool start parts = %+v", parts)
	}

	parts, err = decoder.Decode(RawStreamEvent{Event: "response.function_call_arguments.delta", Data: []byte(`{"type":"response.function_call_arguments.delta","item_id":"fc_1","delta":"{\"city\":"}`)})
	if err != nil {
		t.Fatalf("Decode(tool delta) error = %v", err)
	}
	if len(parts) != 1 || parts[0].Type != StreamToolInputDelta || parts[0].ID != "fc_1" || parts[0].ToolCallID != "call_1" || parts[0].ToolName != "get_weather" || parts[0].Delta != `{"city":` {
		t.Fatalf("tool delta parts = %+v", parts)
	}

	parts, err = decoder.Decode(RawStreamEvent{Event: "response.function_call_arguments.done", Data: []byte(`{"type":"response.function_call_arguments.done","item_id":"fc_1"}`)})
	if err != nil {
		t.Fatalf("Decode(tool done) error = %v", err)
	}
	if len(parts) != 1 || parts[0].Type != StreamToolInputEnd || parts[0].ID != "fc_1" || parts[0].ToolCallID != "call_1" || parts[0].ToolName != "get_weather" {
		t.Fatalf("tool done parts = %+v", parts)
	}

	parts, err = decoder.Decode(RawStreamEvent{Event: "response.output_item.done", Data: []byte(`{"type":"response.output_item.done","item":{"id":"fc_1","type":"function_call","name":"get_weather","call_id":"call_1","status":"completed"}}`)})
	if err != nil {
		t.Fatalf("Decode(tool item done) error = %v", err)
	}
	if len(parts) != 0 {
		t.Fatalf("duplicate tool item done parts = %+v", parts)
	}

	parts, err = decoder.Decode(RawStreamEvent{Event: "response.completed", Data: []byte(`{"type":"response.completed","response":{"id":"resp_1","status":"completed","usage":{"input_tokens":10,"output_tokens":5}}}`)})
	if err != nil {
		t.Fatalf("Decode(completed) error = %v", err)
	}
	if len(parts) != 1 || parts[0].Type != StreamFinish || parts[0].FinishReason != FinishToolCalls {
		t.Fatalf("completed parts = %+v", parts)
	}
	if parts[0].Usage.InputTokens == nil || *parts[0].Usage.InputTokens != 10 {
		t.Fatalf("completed usage = %+v", parts[0].Usage)
	}

	parts, err = decoder.Decode(RawStreamEvent{Event: "response.incomplete", Data: []byte(`{"type":"response.incomplete","response":{"id":"resp_1","status":"incomplete","incomplete_details":{"reason":"content_filter"},"usage":{"input_tokens":10,"output_tokens":5}}}`)})
	if err != nil {
		t.Fatalf("Decode(incomplete content_filter) error = %v", err)
	}
	if len(parts) != 1 || parts[0].Type != StreamFinish || parts[0].FinishReason != FinishContentFilter {
		t.Fatalf("incomplete content_filter parts = %+v", parts)
	}

	parts, err = decoder.Decode(RawStreamEvent{Event: "response.incomplete", Data: []byte(`{"type":"response.incomplete","response":{"id":"resp_1","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"}}}`)})
	if err != nil {
		t.Fatalf("Decode(incomplete max_output_tokens) error = %v", err)
	}
	if len(parts) != 1 || parts[0].Type != StreamFinish || parts[0].FinishReason != FinishLength {
		t.Fatalf("incomplete max_output_tokens parts = %+v", parts)
	}

	parts, err = decoder.Decode(RawStreamEvent{Event: "response.output_item.added", Data: []byte(`{"type":"response.output_item.added","item":{"id":"custom_1","type":"custom_tool_call","name":"grammar_tool","call_id":"call_custom","status":"in_progress"}}`)})
	if err != nil {
		t.Fatalf("Decode(custom tool start) error = %v", err)
	}
	if len(parts) != 1 || parts[0].Type != StreamToolInputStart || parts[0].ToolCallID != "call_custom" || parts[0].ProviderMetadata["custom_tool_call"] != true {
		t.Fatalf("custom tool start parts = %+v", parts)
	}

	parts, err = decoder.Decode(RawStreamEvent{Event: "response.custom_tool_call_input.delta", Data: []byte(`{"type":"response.custom_tool_call_input.delta","item_id":"custom_1","delta":"raw"}`)})
	if err != nil {
		t.Fatalf("Decode(custom tool delta) error = %v", err)
	}
	if len(parts) != 1 || parts[0].Type != StreamToolInputDelta || parts[0].ID != "custom_1" || parts[0].ToolCallID != "call_custom" || parts[0].ToolName != "grammar_tool" || parts[0].Delta != "raw" || parts[0].ProviderMetadata["custom_tool_call"] != true {
		t.Fatalf("custom tool delta parts = %+v", parts)
	}
}

func TestOpenAIResponsesStreamDecoderMergesImageGenerationUsage(t *testing.T) {
	decoder, err := NewOpenAIResponsesAdapter().NewStreamDecoder(StreamDecodeOptions{})
	if err != nil {
		t.Fatalf("NewStreamDecoder() error = %v", err)
	}

	parts, err := decoder.Decode(RawStreamEvent{Event: "response.completed", Data: []byte(`{
		"type":"response.completed",
		"response":{
			"id":"resp_1",
			"status":"completed",
			"usage":{"input_tokens":10,"output_tokens":3,"input_tokens_details":{"cached_tokens":4}},
			"output":[{
				"type":"image_generation_call",
				"usage":{
					"input_tokens":7,
					"output_tokens":11,
					"input_tokens_details":{"text_tokens":2,"image_tokens":5}
				}
			}]
		}
	}`)})
	if err != nil {
		t.Fatalf("Decode(completed) error = %v", err)
	}
	if len(parts) != 1 || parts[0].Type != StreamFinish {
		t.Fatalf("completed parts = %+v", parts)
	}
	billingUsage := billingUsageForProtocol(ProtocolOpenAIResponses, parts[0].Usage)
	if billingUsage.InputTokens != 13 || billingUsage.CachedInputTokens != 4 || billingUsage.OutputTokens != 14 {
		t.Fatalf("billing usage = %+v", billingUsage)
	}
}

func TestOpenAIResponsesStreamDecoderKeepsImageGenerationUsageFromOutputItemDone(t *testing.T) {
	decoder, err := NewOpenAIResponsesAdapter().NewStreamDecoder(StreamDecodeOptions{})
	if err != nil {
		t.Fatalf("NewStreamDecoder() error = %v", err)
	}

	parts, err := decoder.Decode(RawStreamEvent{Event: "response.output_item.done", Data: []byte(`{
		"type":"response.output_item.done",
		"output_index":0,
		"item":{
			"id":"ig_1",
			"type":"image_generation_call",
			"usage":{
				"input_tokens_details":{"text_tokens":2,"image_tokens":5},
				"output_tokens_details":{"image_tokens":11}
			}
		}
	}`)})
	if err != nil {
		t.Fatalf("Decode(output item done) error = %v", err)
	}
	if len(parts) != 0 {
		t.Fatalf("output item done parts = %+v", parts)
	}

	parts, err = decoder.Decode(RawStreamEvent{Event: "response.completed", Data: []byte(`{
		"type":"response.completed",
		"response":{
			"id":"resp_1",
			"status":"completed",
			"usage":{"input_tokens":10,"output_tokens":3,"input_tokens_details":{"cached_tokens":4}}
		}
	}`)})
	if err != nil {
		t.Fatalf("Decode(completed) error = %v", err)
	}
	if len(parts) != 1 || parts[0].Type != StreamFinish {
		t.Fatalf("completed parts = %+v", parts)
	}
	billingUsage := billingUsageForProtocol(ProtocolOpenAIResponses, parts[0].Usage)
	if billingUsage.InputTokens != 13 || billingUsage.CachedInputTokens != 4 || billingUsage.OutputTokens != 14 {
		t.Fatalf("billing usage = %+v", billingUsage)
	}
}

func TestOpenAIResponsesStreamDecoderDoesNotDoubleCountRememberedImageGenerationUsage(t *testing.T) {
	decoder, err := NewOpenAIResponsesAdapter().NewStreamDecoder(StreamDecodeOptions{})
	if err != nil {
		t.Fatalf("NewStreamDecoder() error = %v", err)
	}

	if _, err := decoder.Decode(RawStreamEvent{Event: "response.output_item.done", Data: []byte(`{
		"type":"response.output_item.done",
		"output_index":0,
		"item":{"id":"ig_1","type":"image_generation_call","usage":{"input_tokens":7,"output_tokens":11}}
	}`)}); err != nil {
		t.Fatalf("Decode(output item done) error = %v", err)
	}

	parts, err := decoder.Decode(RawStreamEvent{Event: "response.completed", Data: []byte(`{
		"type":"response.completed",
		"response":{
			"id":"resp_1",
			"status":"completed",
			"usage":{"input_tokens":10,"output_tokens":3,"input_tokens_details":{"cached_tokens":4}},
			"output":[{"id":"ig_1","type":"image_generation_call","usage":{"input_tokens":7,"output_tokens":11}}]
		}
	}`)})
	if err != nil {
		t.Fatalf("Decode(completed) error = %v", err)
	}
	if len(parts) != 1 || parts[0].Type != StreamFinish {
		t.Fatalf("completed parts = %+v", parts)
	}
	billingUsage := billingUsageForProtocol(ProtocolOpenAIResponses, parts[0].Usage)
	if billingUsage.InputTokens != 13 || billingUsage.CachedInputTokens != 4 || billingUsage.OutputTokens != 14 {
		t.Fatalf("billing usage = %+v", billingUsage)
	}
}

func TestOpenAIResponsesStreamDecoderDoesNotDoubleCountDuplicateImageGenerationOutput(t *testing.T) {
	decoder, err := NewOpenAIResponsesAdapter().NewStreamDecoder(StreamDecodeOptions{})
	if err != nil {
		t.Fatalf("NewStreamDecoder() error = %v", err)
	}

	parts, err := decoder.Decode(RawStreamEvent{Event: "response.completed", Data: []byte(`{
		"type":"response.completed",
		"response":{
			"id":"resp_1",
			"status":"completed",
			"usage":{"input_tokens":10,"output_tokens":3,"input_tokens_details":{"cached_tokens":4}},
			"output":[{
				"id":"ig_1",
				"type":"image_generation_call",
				"usage":{"input_tokens":7,"output_tokens":11}
			}]
		},
		"output":[{
			"id":"ig_1",
			"type":"image_generation_call",
			"usage":{"input_tokens":7,"output_tokens":11}
		}]
	}`)})
	if err != nil {
		t.Fatalf("Decode(completed) error = %v", err)
	}
	if len(parts) != 1 || parts[0].Type != StreamFinish {
		t.Fatalf("completed parts = %+v", parts)
	}
	billingUsage := billingUsageForProtocol(ProtocolOpenAIResponses, parts[0].Usage)
	if billingUsage.InputTokens != 13 || billingUsage.CachedInputTokens != 4 || billingUsage.OutputTokens != 14 {
		t.Fatalf("billing usage = %+v", billingUsage)
	}
}

func TestOpenAIResponsesStreamEncoder(t *testing.T) {
	encoder, err := NewOpenAIResponsesAdapter().NewStreamEncoder(StreamEncodeOptions{Model: "gpt-4"})
	if err != nil {
		t.Fatalf("NewStreamEncoder() error = %v", err)
	}

	events, err := encoder.Encode(StreamPart{Type: StreamStart, ID: "resp_1"})
	if err != nil {
		t.Fatalf("Encode(StreamStart) error = %v", err)
	}
	if len(events) != 2 || events[0].Event != "response.created" || events[1].Event != "response.in_progress" {
		t.Fatalf("StreamStart events = %+v", events)
	}

	if _, err := encoder.Encode(StreamPart{Type: StreamTextStart, ID: "msg_1"}); err != nil {
		t.Fatalf("Encode(StreamTextStart) error = %v", err)
	}
	events, err = encoder.Encode(StreamPart{Type: StreamTextDelta, ID: "msg_1", Delta: "Hi"})
	if err != nil {
		t.Fatalf("Encode(StreamTextDelta) error = %v", err)
	}
	if len(events) != 1 || events[0].Event != "response.output_text.delta" {
		t.Fatalf("StreamTextDelta events = %+v", events)
	}
	var textDelta openAIResponsesStreamEvent
	if err := json.Unmarshal(events[0].Data, &textDelta); err != nil {
		t.Fatalf("Unmarshal(text delta) error = %v", err)
	}
	if textDelta.ItemID == "" || textDelta.ContentIndex == nil || *textDelta.ContentIndex != 0 {
		t.Fatalf("text delta missing indexes = %+v", textDelta)
	}
	if _, err := encoder.Encode(StreamPart{Type: StreamTextEnd, ID: "msg_1"}); err != nil {
		t.Fatalf("Encode(StreamTextEnd) error = %v", err)
	}

	if _, err := encoder.Encode(StreamPart{Type: StreamReasoningStart, ID: "rs_1"}); err != nil {
		t.Fatalf("Encode(StreamReasoningStart) error = %v", err)
	}
	if _, err := encoder.Encode(StreamPart{Type: StreamReasoningDelta, Delta: "summary"}); err != nil {
		t.Fatalf("Encode(StreamReasoningDelta) error = %v", err)
	}
	if events, err := encoder.Encode(StreamPart{Type: StreamReasoningDelta, ProviderMetadata: map[string]any{"signature": "enc_1"}}); err != nil || len(events) != 0 {
		t.Fatalf("Encode(StreamReasoningSignature) events = %+v err = %v", events, err)
	}
	reasonDone, err := encoder.Encode(StreamPart{Type: StreamReasoningEnd})
	if err != nil {
		t.Fatalf("Encode(StreamReasoningEnd) error = %v", err)
	}
	var reasonEvent openAIResponsesStreamEvent
	if err := json.Unmarshal(reasonDone[0].Data, &reasonEvent); err != nil {
		t.Fatalf("Unmarshal(reasonDone) error = %v", err)
	}
	if reasonEvent.Item == nil || reasonEvent.Item.EncryptedContent != "enc_1" {
		t.Fatalf("reasonDone = %+v", reasonEvent)
	}

	customEvents, err := encoder.Encode(StreamPart{Type: StreamToolInputStart, ToolCallID: "call_custom", ToolName: "grammar_tool", ProviderMetadata: map[string]any{"custom_tool_call": true}})
	if err != nil {
		t.Fatalf("Encode(custom start) error = %v", err)
	}
	var customStart openAIResponsesStreamEvent
	if err := json.Unmarshal(customEvents[0].Data, &customStart); err != nil {
		t.Fatalf("Unmarshal(custom start) error = %v", err)
	}
	if customStart.Item == nil || customStart.Item.Type != "custom_tool_call" || customStart.Item.Input != "" {
		t.Fatalf("customStart = %+v", customStart)
	}
	if _, err := encoder.Encode(StreamPart{Type: StreamToolInputDelta, Delta: "raw", ProviderMetadata: map[string]any{"custom_tool_call": true}}); err != nil {
		t.Fatalf("Encode(custom delta) error = %v", err)
	}
	customDone, err := encoder.Encode(StreamPart{Type: StreamToolInputEnd, ProviderMetadata: map[string]any{"custom_tool_call": true}})
	if err != nil {
		t.Fatalf("Encode(custom end) error = %v", err)
	}
	var customDoneEvent openAIResponsesStreamEvent
	if err := json.Unmarshal(customDone[0].Data, &customDoneEvent); err != nil {
		t.Fatalf("Unmarshal(custom done) error = %v", err)
	}
	if customDoneEvent.ItemID == "" {
		t.Fatalf("custom done missing item_id = %+v", customDoneEvent)
	}

	events, err = encoder.Encode(StreamPart{Type: StreamFinish, FinishReason: FinishStop})
	if err != nil {
		t.Fatalf("Encode(StreamFinish) error = %v", err)
	}
	if len(events) != 1 || events[0].Event != "response.completed" {
		t.Fatalf("StreamFinish events = %+v", events)
	}

	events, err = encoder.Close()
	if err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if len(events) != 0 {
		t.Fatalf("Close events should be empty = %+v", events)
	}
}

func TestOpenAIResponsesStreamEncoderIncludesActiveItemIndexes(t *testing.T) {
	encoder, err := NewOpenAIResponsesAdapter().NewStreamEncoder(StreamEncodeOptions{Model: "gpt-4"})
	if err != nil {
		t.Fatalf("NewStreamEncoder() error = %v", err)
	}
	if _, err := encoder.Encode(StreamPart{Type: StreamStart, ID: "resp_1"}); err != nil {
		t.Fatalf("Encode(StreamStart) error = %v", err)
	}

	reasonStart, err := encoder.Encode(StreamPart{Type: StreamReasoningStart, ID: "rs_1"})
	if err != nil {
		t.Fatalf("Encode(StreamReasoningStart) error = %v", err)
	}
	reasonStartPayload := rawStreamEventMap(t, reasonStart[0])
	if _, ok := reasonStartPayload["output_index"]; !ok || reasonStartPayload["output_index"] != float64(0) {
		t.Fatalf("reasoning start missing output_index 0: %+v", reasonStartPayload)
	}
	if _, err := encoder.Encode(StreamPart{Type: StreamReasoningEnd}); err != nil {
		t.Fatalf("Encode(StreamReasoningEnd) error = %v", err)
	}

	textStart, err := encoder.Encode(StreamPart{Type: StreamTextStart, ID: "msg_1"})
	if err != nil {
		t.Fatalf("Encode(StreamTextStart) error = %v", err)
	}
	itemAddedPayload := rawStreamEventMap(t, textStart[0])
	if _, ok := itemAddedPayload["output_index"]; !ok || itemAddedPayload["output_index"] != float64(1) {
		t.Fatalf("text item added missing output_index 1: %+v", itemAddedPayload)
	}
	contentAddedPayload := rawStreamEventMap(t, textStart[1])
	if _, ok := contentAddedPayload["output_index"]; !ok || contentAddedPayload["output_index"] != float64(1) {
		t.Fatalf("content added missing output_index 1: %+v", contentAddedPayload)
	}
	if _, ok := contentAddedPayload["content_index"]; !ok || contentAddedPayload["content_index"] != float64(0) {
		t.Fatalf("content added missing content_index 0: %+v", contentAddedPayload)
	}

	textDelta, err := encoder.Encode(StreamPart{Type: StreamTextDelta, ID: "msg_1", Delta: "Hi"})
	if err != nil {
		t.Fatalf("Encode(StreamTextDelta) error = %v", err)
	}
	textDeltaPayload := rawStreamEventMap(t, textDelta[0])
	if _, ok := textDeltaPayload["output_index"]; !ok || textDeltaPayload["output_index"] != float64(1) {
		t.Fatalf("text delta missing output_index 1: %+v", textDeltaPayload)
	}
	if _, ok := textDeltaPayload["content_index"]; !ok || textDeltaPayload["content_index"] != float64(0) {
		t.Fatalf("text delta missing content_index 0: %+v", textDeltaPayload)
	}
}

func TestOpenAIResponsesStreamEncoderStartsTextItemForInitialDelta(t *testing.T) {
	encoder, err := NewOpenAIResponsesAdapter().NewStreamEncoder(StreamEncodeOptions{Model: "gpt-4"})
	if err != nil {
		t.Fatalf("NewStreamEncoder() error = %v", err)
	}
	if _, err := encoder.Encode(StreamPart{Type: StreamStart, ID: "resp_1"}); err != nil {
		t.Fatalf("Encode(StreamStart) error = %v", err)
	}

	events, err := encoder.Encode(StreamPart{Type: StreamTextDelta, Delta: "Hi"})
	if err != nil {
		t.Fatalf("Encode(StreamTextDelta) error = %v", err)
	}
	if len(events) != 3 || events[0].Event != "response.output_item.added" || events[1].Event != "response.content_part.added" || events[2].Event != "response.output_text.delta" {
		t.Fatalf("StreamTextDelta should start an item before delta, events = %+v", events)
	}

	itemAddedPayload := rawStreamEventMap(t, events[0])
	if _, ok := itemAddedPayload["output_index"]; !ok || itemAddedPayload["output_index"] != float64(0) {
		t.Fatalf("item added missing output_index 0: %+v", itemAddedPayload)
	}
	contentAddedPayload := rawStreamEventMap(t, events[1])
	if _, ok := contentAddedPayload["content_index"]; !ok || contentAddedPayload["content_index"] != float64(0) {
		t.Fatalf("content added missing content_index 0: %+v", contentAddedPayload)
	}
	textDeltaPayload := rawStreamEventMap(t, events[2])
	if _, ok := textDeltaPayload["output_index"]; !ok || textDeltaPayload["output_index"] != float64(0) {
		t.Fatalf("text delta missing output_index 0: %+v", textDeltaPayload)
	}
	if _, ok := textDeltaPayload["content_index"]; !ok || textDeltaPayload["content_index"] != float64(0) {
		t.Fatalf("text delta missing content_index 0: %+v", textDeltaPayload)
	}
}

func rawStreamEventMap(t *testing.T, event RawStreamEvent) map[string]any {
	t.Helper()
	var payload map[string]any
	if err := json.Unmarshal(event.Data, &payload); err != nil {
		t.Fatalf("Unmarshal(%s) error = %v", event.Event, err)
	}
	return payload
}

func TestOpenAIResponsesStreamEncoderFinishDetails(t *testing.T) {
	finishCases := []struct {
		name       string
		part       StreamPart
		eventType  string
		status     string
		reason     string
		wantErrVal bool
	}{
		{name: "length", part: StreamPart{Type: StreamFinish, FinishReason: FinishLength}, eventType: "response.incomplete", status: "incomplete", reason: "max_output_tokens"},
		{name: "content filter", part: StreamPart{Type: StreamFinish, FinishReason: FinishContentFilter}, eventType: "response.incomplete", status: "incomplete", reason: "content_filter"},
		{name: "error", part: StreamPart{Type: StreamFinish, FinishReason: FinishError, Error: map[string]any{"message": "upstream failed"}}, eventType: "response.failed", status: "failed", wantErrVal: true},
	}

	for _, tc := range finishCases {
		t.Run(tc.name, func(t *testing.T) {
			encoder, err := NewOpenAIResponsesAdapter().NewStreamEncoder(StreamEncodeOptions{Model: "gpt-4"})
			if err != nil {
				t.Fatalf("NewStreamEncoder() error = %v", err)
			}
			events, err := encoder.Encode(tc.part)
			if err != nil {
				t.Fatalf("Encode(StreamFinish) error = %v", err)
			}
			if len(events) != 1 || events[0].Event != tc.eventType {
				t.Fatalf("events = %+v", events)
			}

			var event openAIResponsesStreamEvent
			if err := json.Unmarshal(events[0].Data, &event); err != nil {
				t.Fatalf("Unmarshal(finish) error = %v", err)
			}
			if event.Type != tc.eventType || event.Response == nil || event.Response.Status != tc.status {
				t.Fatalf("finish event = %+v", event)
			}
			if tc.reason != "" {
				if event.Response.IncompleteDetails == nil || event.Response.IncompleteDetails.Reason != tc.reason {
					t.Fatalf("incomplete_details = %+v", event.Response.IncompleteDetails)
				}
			}
			if tc.wantErrVal && event.Response.Error == nil {
				t.Fatalf("error missing from response = %+v", event.Response)
			}
		})
	}
}
