package protocolbridge

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sort"
	"strings"
)

type OpenAIResponsesAdapter struct{}

func NewOpenAIResponsesAdapter() OpenAIResponsesAdapter {
	return OpenAIResponsesAdapter{}
}

func (a OpenAIResponsesAdapter) Protocol() Protocol {
	return ProtocolOpenAIResponses
}

func (a OpenAIResponsesAdapter) DecodeRequest(raw []byte) (*LLMRequest, error) {
	var request openAIResponsesRequest
	if err := json.Unmarshal(raw, &request); err != nil {
		return nil, fmt.Errorf("decode openai responses request: %w", err)
	}

	prompt := make([]Message, 0)
	if strings.TrimSpace(request.Instructions) != "" {
		prompt = append(prompt, Message{Role: RoleSystem, Parts: []Part{{Type: PartText, Text: &TextPart{Text: request.Instructions}}}})
	}
	inputMessages, err := decodeOpenAIResponsesInput(asRawMessage(request.Input))
	if err != nil {
		return nil, err
	}
	prompt = append(prompt, inputMessages...)

	reasoning, reasoningEffort, reasoningSummary := decodeOpenAIResponsesReasoningDetails(request.Reasoning)

	return &LLMRequest{
		Protocol:          ProtocolOpenAIResponses,
		Model:             request.Model,
		Prompt:            prompt,
		MaxOutputTokens:   maxOutputTokensOrDefault(request.MaxOutputTokens),
		Temperature:       request.Temperature,
		TopP:              request.TopP,
		ResponseFormat:    decodeOpenAIResponsesTextConfig(request.Text),
		Reasoning:         reasoning,
		ReasoningEffort:   reasoningEffort,
		ReasoningSummary:  reasoningSummary,
		Tools:             decodeOpenAIResponsesTools(request.Tools),
		ToolChoice:        decodeOpenAIResponseToolChoice(asRawMessage(request.ToolChoice)),
		State:             decodeOpenAIResponsesState(request),
		Include:           request.Include,
		ParallelToolCalls: request.ParallelToolCalls,
		Stream:            request.Stream,
	}, nil
}

func (a OpenAIResponsesAdapter) EncodeRequest(req *LLMRequest, opts EncodeRequestOptions) ([]byte, error) {
	if req == nil {
		return nil, errors.New("encode openai responses request: nil request")
	}
	if len(req.StopSequences) > 0 {
		return nil, errors.New("encode openai responses request: stop sequences are not supported")
	}

	model := req.Model
	if opts.Model != "" {
		model = opts.Model
	}

	request := openAIResponsesRequest{
		Model:             model,
		MaxOutputTokens:   maxOutputTokensOrDefault(req.MaxOutputTokens),
		Temperature:       req.Temperature,
		TopP:              req.TopP,
		Text:              encodeOpenAIResponsesTextConfig(req.ResponseFormat),
		Reasoning:         encodeOpenAIResponsesReasoningConfig(req),
		Tools:             encodeOpenAIResponsesTools(req.Tools),
		ToolChoice:        encodeOpenAIResponsesToolChoice(req.ToolChoice),
		Include:           req.Include,
		ParallelToolCalls: req.ParallelToolCalls,
		Stream:            req.Stream,
	}
	encodeOpenAIResponsesState(&request, req.State)
	input := make([]openAIResponsesInputItem, 0)
	for _, message := range req.Prompt {
		if message.Role == RoleSystem || message.Role == RoleDeveloper {
			request.Instructions = appendInstructionsText(request.Instructions, joinTextParts(message.Parts))
			continue
		}
		if message.Role == RoleTool {
			input = append(input, encodeOpenAIResponsesToolResults(message.Parts)...)
			continue
		}
		input = append(input, openAIResponsesInputItem{
			Role:    string(message.Role),
			Content: encodeOpenAIResponsesInputContent(message.Role, message.Parts),
		})
		if message.Role == RoleAssistant {
			input = append(input, encodeOpenAIResponsesToolCalls(message.Parts)...)
		}
	}
	request.Input = input

	return json.Marshal(request)
}

func appendInstructionsText(instructions string, text string) string {
	if strings.TrimSpace(text) == "" {
		return instructions
	}
	if strings.TrimSpace(instructions) == "" {
		return text
	}
	return instructions + "\n" + text
}

func (a OpenAIResponsesAdapter) DecodeResponse(raw []byte) (*LLMResponse, error) {
	var response openAIResponsesResponse
	if err := json.Unmarshal(raw, &response); err != nil {
		return nil, fmt.Errorf("decode openai responses response: %w", err)
	}

	content := make([]Part, 0)
	hasToolCall := false
	hasRefusal := false
	for _, item := range response.Output {
		if item.Type == "reasoning" {
			content = append(content, decodeOpenAIResponsesReasoning(item)...)
			continue
		}
		if item.Type == "output_text" {
			content = append(content, decodeOpenAIResponsesOutputTextItem(item)...)
			continue
		}
		if item.Type == "function_call" || item.Type == "custom_tool_call" {
			hasToolCall = true
			toolCall, err := decodeOpenAIResponsesToolCall(item)
			if err != nil {
				return nil, err
			}
			content = append(content, toolCall)
			continue
		}
		if item.Type == "function_call_output" {
			content = append(content, decodeOpenAIResponsesToolResult(item))
			continue
		}
		if item.Type == "image_generation_call" || item.Type == "input_image" {
			content = append(content, decodeOpenAIResponsesImageOutputItem(item)...)
			continue
		}
		if item.Type != "message" {
			continue
		}
		decodedContent := decodeOpenAIResponsesOutputContent(item.Content)
		if hasRefusalPart(decodedContent) {
			hasRefusal = true
		}
		content = append(content, decodedContent...)
	}
	if len(content) == 0 && response.OutputText != "" {
		content = append(content, Part{Type: PartText, Text: &TextPart{Text: response.OutputText}})
	}
	finishReason := decodeOpenAIResponsesFinishReason(response.Status, response.IncompleteDetails)
	if hasRefusal {
		finishReason = FinishContentFilter
	} else if finishReason == FinishStop && hasToolCall {
		finishReason = FinishToolCalls
	}

	return &LLMResponse{
		Protocol:         ProtocolOpenAIResponses,
		ID:               response.ID,
		Model:            response.Model,
		Role:             RoleAssistant,
		Content:          content,
		FinishReason:     finishReason,
		Usage:            decodeOpenAIResponsesUsage(response.Usage),
		ProviderMetadata: map[string]any{"object": response.Object, "status": response.Status},
	}, nil
}

func (a OpenAIResponsesAdapter) EncodeResponse(resp *LLMResponse, opts EncodeResponseOptions) ([]byte, error) {
	if resp == nil {
		return nil, errors.New("encode openai responses response: nil response")
	}

	model := resp.Model
	if opts.Model != "" {
		model = opts.Model
	}

	content, finishReason := firstResponseContent(resp)
	response := openAIResponsesResponse{
		ID:         resp.ID,
		Object:     "response",
		Status:     encodeOpenAIResponsesStatus(finishReason),
		Model:      model,
		OutputText: joinTextParts(content),
		Usage:      encodeOpenAIResponsesUsage(resp.Usage, resp.BillingUsage()),
	}
	if item, ok := encodeOpenAIResponsesReasoningOutputItem(content, prefixedID(resp.ID, "rs"), encodeOpenAIResponsesStatus(finishReason)); ok {
		response.Output = append(response.Output, item)
	}
	if outputContent := encodeOpenAIResponsesOutputContent(content); len(outputContent) > 0 {
		response.Output = append(response.Output, openAIResponsesOutputItem{ID: prefixedID(resp.ID, "msg"), Type: "message", Role: string(RoleAssistant), Status: encodeOpenAIResponsesStatus(finishReason), Content: outputContent})
	}
	response.Output = append(response.Output, encodeOpenAIResponsesResponseToolCalls(content, finishReason)...)
	response.Output = append(response.Output, encodeOpenAIResponsesResponseToolResults(content, finishReason)...)

	return json.Marshal(response)
}

func (a OpenAIResponsesAdapter) NewStreamDecoder(StreamDecodeOptions) (StreamDecoder, error) {
	return &openAIResponsesStreamDecoder{}, nil
}

func (a OpenAIResponsesAdapter) NewStreamEncoder(opts StreamEncodeOptions) (StreamEncoder, error) {
	return &openAIResponsesStreamEncoder{model: opts.Model}, nil
}

func (a OpenAIResponsesAdapter) EncodeError(err error) ([]byte, int) {
	message := "unknown error"
	if err != nil {
		message = err.Error()
	}
	raw, marshalErr := json.Marshal(openAIErrorResponse{Error: openAIError{Message: message, Type: "protocol_bridge_error"}})
	if marshalErr != nil {
		return []byte(`{"error":{"message":"failed to encode error","type":"protocol_bridge_error"}}`), http.StatusInternalServerError
	}
	return raw, http.StatusBadRequest
}

func decodeOpenAIResponsesInput(raw json.RawMessage) ([]Message, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}

	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return []Message{{Role: RoleUser, Parts: []Part{{Type: PartText, Text: &TextPart{Text: text}}}}}, nil
	}

	var messages []openAIResponsesInputItem
	if err := json.Unmarshal(raw, &messages); err != nil {
		return nil, fmt.Errorf("decode openai responses input: %w", err)
	}

	decoded := make([]Message, 0, len(messages))
	for _, message := range messages {
		if message.Type == "function_call" {
			toolCall, err := decodeOpenAIResponsesToolCall(openAIResponsesOutputItemFromInputItem(message))
			if err != nil {
				return nil, err
			}
			decoded = append(decoded, Message{Role: RoleAssistant, Parts: []Part{toolCall}})
			continue
		}
		if message.Type == "custom_tool_call" {
			toolCall, err := decodeOpenAIResponsesToolCall(openAIResponsesOutputItemFromInputItem(message))
			if err != nil {
				return nil, err
			}
			decoded = append(decoded, Message{Role: RoleAssistant, Parts: []Part{toolCall}})
			continue
		}
		if message.Type == "function_call_output" {
			decoded = append(decoded, Message{Role: RoleTool, Parts: []Part{decodeOpenAIResponsesToolResult(openAIResponsesOutputItemFromInputItem(message))}})
			continue
		}
		parts, err := decodeOpenAIResponsesContent(asRawMessage(message.Content))
		if err != nil {
			return nil, err
		}
		if message.Role == "" && len(parts) == 0 {
			continue
		}
		decoded = append(decoded, Message{Role: Role(message.Role), Parts: parts})
	}
	return decoded, nil
}

func decodeOpenAIResponsesContent(raw json.RawMessage) ([]Part, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return []Part{{Type: PartText, Text: &TextPart{Text: text}}}, nil
	}
	var parts []openAIResponsesContentPart
	if err := json.Unmarshal(raw, &parts); err != nil {
		return nil, fmt.Errorf("decode openai responses content: %w", err)
	}
	decoded := make([]Part, 0, len(parts))
	for _, part := range parts {
		if part.Type == "input_text" || part.Type == "output_text" || part.Type == "text" {
			decoded = append(decoded, Part{Type: PartText, Text: &TextPart{Text: part.Text}})
			continue
		}
		if part.Type == "input_image" {
			file := &FilePart{Type: FileImage, FileID: part.FileID, Detail: part.Detail}
			if part.ImageURL != "" {
				file = decodeFileURL(part.ImageURL, FileImage)
				file.FileID = part.FileID
				file.Detail = part.Detail
			}
			decoded = append(decoded, Part{Type: PartFile, File: file})
			continue
		}
		if part.Type == "input_file" {
			file := &FilePart{Type: FileDocument, Data: part.FileData, URL: part.FileURL, FileID: part.FileID, Filename: part.Filename, Detail: part.Detail}
			decoded = append(decoded, Part{Type: PartFile, File: file})
		}
	}
	return decoded, nil
}

func decodeOpenAIResponsesReasoningConfig(raw any) *bool {
	if len(asRawMessage(raw)) == 0 || string(asRawMessage(raw)) == "null" {
		return nil
	}
	var config struct {
		Effort string `json:"effort"`
	}
	if err := json.Unmarshal(asRawMessage(raw), &config); err == nil && strings.TrimSpace(config.Effort) == "none" {
		enabled := false
		return &enabled
	}
	enabled := true
	return &enabled
}

func decodeOpenAIResponsesReasoningDetails(raw any) (*bool, string, string) {
	if len(asRawMessage(raw)) == 0 || string(asRawMessage(raw)) == "null" {
		return nil, "", ""
	}
	var config struct {
		Effort  string `json:"effort"`
		Summary string `json:"summary"`
	}
	if err := json.Unmarshal(asRawMessage(raw), &config); err != nil {
		enabled := true
		return &enabled, "", ""
	}
	effort := strings.TrimSpace(config.Effort)
	summary := strings.TrimSpace(config.Summary)
	if effort == "none" {
		enabled := false
		return &enabled, effort, summary
	}
	enabled := true
	return &enabled, effort, summary
}

func encodeOpenAIResponsesReasoningConfig(req *LLMRequest) any {
	if req == nil {
		return nil
	}
	effort := strings.TrimSpace(req.ReasoningEffort)
	if effort == "none" {
		return nil
	}
	config := map[string]any{}
	if effort != "" {
		config["effort"] = effort
	} else if req.ReasoningBudgetTokens != nil {
		config["effort"] = mapReasoningBudgetToOpenAIEffort(*req.ReasoningBudgetTokens)
	} else if req.Reasoning != nil && *req.Reasoning {
		config["effort"] = "medium"
	}
	if strings.TrimSpace(req.ReasoningSummary) != "" {
		config["summary"] = req.ReasoningSummary
	}
	if len(config) == 0 {
		return nil
	}
	return config
}

func mapReasoningBudgetToOpenAIEffort(budget int) string {
	switch {
	case budget >= 8192:
		return "xhigh"
	case budget >= 4096:
		return "high"
	default:
		return "medium"
	}
}

func encodeOpenAIResponsesContent(parts []Part) []openAIResponsesContentPart {
	return encodeOpenAIResponsesInputContent(RoleUser, parts)
}

func encodeOpenAIResponsesInputContent(role Role, parts []Part) []openAIResponsesContentPart {
	encoded := make([]openAIResponsesContentPart, 0, len(parts))
	textType := "input_text"
	if role == RoleAssistant {
		textType = "output_text"
	}
	for _, part := range parts {
		if part.Type == PartText && part.Text != nil {
			encoded = append(encoded, openAIResponsesContentPart{Type: textType, Text: part.Text.Text})
			continue
		}
		if role == RoleUser && part.Type == PartFile && part.File != nil && part.File.Type == FileImage {
			detail := part.File.Detail
			if detail == "" {
				detail = "auto"
			}
			image := openAIResponsesContentPart{Type: "input_image", Detail: detail}
			if url := encodeFileURL(part.File); url != "" {
				image.ImageURL = url
			} else {
				image.FileID = part.File.FileID
			}
			if image.ImageURL == "" && image.FileID == "" {
				continue
			}
			encoded = append(encoded, image)
			continue
		}
		if role == RoleUser && part.Type == PartFile && part.File != nil && part.File.Type == FileDocument {
			file := openAIResponsesContentPart{Type: "input_file", FileData: part.File.Data, FileURL: part.File.URL, FileID: part.File.FileID, Filename: part.File.Filename, Detail: part.File.Detail}
			if file.FileData == "" && file.FileURL == "" && file.FileID == "" {
				continue
			}
			encoded = append(encoded, file)
		}
	}
	return encoded
}

func decodeOpenAIResponsesOutputContent(parts []openAIResponsesContentPart) []Part {
	decoded := make([]Part, 0, len(parts))
	for _, part := range parts {
		if part.Type == "reasoning_text" || part.Type == "summary_text" {
			decoded = append(decoded, Part{Type: PartReasoning, Reasoning: &ReasoningPart{Text: part.Text}})
			continue
		}
		if part.Type == "output_text" || part.Type == "text" {
			decoded = append(decoded, Part{Type: PartText, Text: &TextPart{Text: part.Text}})
			continue
		}
		if part.Type == "refusal" {
			decoded = append(decoded, Part{Type: PartRefusal, Refusal: &RefusalPart{Text: part.Refusal}})
		}
	}
	return decoded
}

func hasRefusalPart(parts []Part) bool {
	for _, part := range parts {
		if part.Type == PartRefusal && part.Refusal != nil {
			return true
		}
	}
	return false
}

func encodeOpenAIResponsesOutputContent(parts []Part) []openAIResponsesContentPart {
	encoded := make([]openAIResponsesContentPart, 0, len(parts))
	for _, part := range parts {
		if part.Type == PartText && part.Text != nil {
			encoded = append(encoded, openAIResponsesContentPart{Type: "output_text", Text: part.Text.Text})
			continue
		}
		if part.Type == PartRefusal && part.Refusal != nil {
			encoded = append(encoded, openAIResponsesContentPart{Type: "refusal", Refusal: part.Refusal.Text})
		}
	}
	return encoded
}

func decodeOpenAIResponsesTextConfig(raw any) *ResponseFormat {
	if len(asRawMessage(raw)) == 0 || string(asRawMessage(raw)) == "null" {
		return nil
	}

	var text struct {
		Format *openAIResponsesTextFormat `json:"format"`
	}
	if err := json.Unmarshal(asRawMessage(raw), &text); err != nil || text.Format == nil {
		return nil
	}

	switch text.Format.Type {
	case "json_object":
		return &ResponseFormat{Type: ResponseFormatJSON}
	case "json_schema":
		return &ResponseFormat{Type: ResponseFormatJSON, Schema: text.Format.Schema, Name: text.Format.Name, Description: text.Format.Description, Strict: text.Format.Strict}
	case "text", "":
		return nil
	default:
		return nil
	}
}

func encodeOpenAIResponsesTextConfig(format *ResponseFormat) any {
	if format == nil || format.Type == ResponseFormatText {
		return nil
	}
	if format.Type != ResponseFormatJSON {
		return nil
	}

	if format.Schema == nil {
		return openAIResponsesTextConfig{Format: openAIResponsesTextFormat{Type: "json_object"}}
	}
	name := format.Name
	if strings.TrimSpace(name) == "" {
		name = "response_format"
	}
	return openAIResponsesTextConfig{Format: openAIResponsesTextFormat{Type: "json_schema", Name: name, Description: format.Description, Schema: format.Schema, Strict: format.Strict}}
}

func decodeOpenAIResponsesState(request openAIResponsesRequest) *RequestState {
	state := &RequestState{PreviousResponseID: request.PreviousResponseID}
	if conversationID, ok := decodeOpenAIResponsesConversationID(asRawMessage(request.Conversation)); ok {
		state.ConversationID = conversationID
	}
	if state.PreviousResponseID == "" && state.ConversationID == "" {
		return nil
	}
	return state
}

func decodeOpenAIResponsesConversationID(raw json.RawMessage) (string, bool) {
	if len(raw) == 0 || string(raw) == "null" {
		return "", false
	}
	var id string
	if err := json.Unmarshal(raw, &id); err == nil {
		return id, id != ""
	}
	var object struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(raw, &object); err == nil {
		return object.ID, object.ID != ""
	}
	return "", false
}

func encodeOpenAIResponsesState(request *openAIResponsesRequest, state *RequestState) {
	if request == nil || state == nil {
		return
	}
	request.PreviousResponseID = state.PreviousResponseID
	if state.ConversationID != "" {
		request.Conversation = state.ConversationID
	}
}

func prefixedID(id string, prefix string) string {
	if id == "" {
		return ""
	}
	return prefix + "_" + id
}

func decodeOpenAIResponsesReasoning(item openAIResponsesOutputItem) []Part {
	decoded := make([]Part, 0, len(item.Summary)+len(item.Content))
	if item.EncryptedContent != "" {
		reasoning := &ReasoningPart{Redacted: item.EncryptedContent, Signature: item.EncryptedContent}
		for _, part := range item.Summary {
			if part.Text != "" {
				reasoning.Text += part.Text
			}
		}
		for _, part := range item.Content {
			if part.Text != "" {
				reasoning.Text += part.Text
			}
		}
		return []Part{{Type: PartReasoning, Reasoning: reasoning}}
	}
	for _, part := range item.Summary {
		if part.Text != "" {
			decoded = append(decoded, Part{Type: PartReasoning, Reasoning: &ReasoningPart{Text: part.Text}})
		}
	}
	for _, part := range item.Content {
		if part.Text != "" {
			decoded = append(decoded, Part{Type: PartReasoning, Reasoning: &ReasoningPart{Text: part.Text}})
		}
	}
	return decoded
}

func decodeOpenAIResponsesOutputTextItem(item openAIResponsesOutputItem) []Part {
	if item.Text == "" {
		return nil
	}
	return []Part{{Type: PartText, Text: &TextPart{Text: item.Text}}}
}

func decodeOpenAIResponsesImageOutputItem(item openAIResponsesOutputItem) []Part {
	file := &FilePart{Type: FileImage}
	if item.ImageURL != "" {
		file = decodeFileURL(item.ImageURL, FileImage)
	}
	if file.URL == "" && file.Data == "" && file.FileID == "" && item.Result != "" {
		mediaType := item.OutputFormat
		if mediaType == "" {
			mediaType = "image/png"
		} else if !strings.Contains(mediaType, "/") {
			mediaType = "image/" + mediaType
		}
		file.MediaType = mediaType
		file.Data = item.Result
	}
	if file.URL == "" && file.Data == "" && file.FileID == "" {
		return nil
	}
	return []Part{{Type: PartFile, File: file}}
}

func encodeOpenAIResponsesReasoningOutputItem(parts []Part, id string, status string) (openAIResponsesOutputItem, bool) {
	item := openAIResponsesOutputItem{ID: id, Type: "reasoning", Status: status}
	for _, part := range parts {
		if part.Type != PartReasoning || part.Reasoning == nil {
			continue
		}
		if strings.TrimSpace(part.Reasoning.Text) != "" {
			item.Summary = append(item.Summary, openAIResponsesContentPart{Type: "summary_text", Text: part.Reasoning.Text})
		}
		if item.EncryptedContent == "" {
			if strings.TrimSpace(part.Reasoning.Redacted) != "" {
				item.EncryptedContent = part.Reasoning.Redacted
			} else if strings.TrimSpace(part.Reasoning.Signature) != "" {
				item.EncryptedContent = part.Reasoning.Signature
			}
		}
	}
	return item, len(item.Summary) > 0 || item.EncryptedContent != ""
}

func decodeOpenAIResponsesTools(tools []openAIResponsesTool) []Tool {
	decoded := make([]Tool, 0, len(tools))
	for _, tool := range tools {
		if tool.Type == "function" {
			decoded = append(decoded, Tool{Type: ToolFunction, Name: tool.Name, Description: tool.Description, InputSchema: tool.Parameters, Strict: tool.Strict})
			continue
		}
		decoded = append(decoded, Tool{Type: ToolProviderDefined, Name: tool.Type, Config: tool.Config()})
	}
	return decoded
}

func encodeOpenAIResponsesTools(tools []Tool) []openAIResponsesTool {
	encoded := make([]openAIResponsesTool, 0, len(tools))
	for _, tool := range tools {
		if tool.Type == ToolFunction {
			encoded = append(encoded, openAIResponsesTool{Type: "function", Name: tool.Name, Description: tool.Description, Parameters: normalizeOpenAIResponsesToolSchema(tool.InputSchema, tool.Strict), Strict: tool.Strict})
			continue
		}
		if tool.Type == ToolProviderDefined && tool.Name != "" {
			encoded = append(encoded, newOpenAIResponsesProviderTool(tool.Name, tool.Config))
		}
	}
	return encoded
}

func normalizeOpenAIResponsesToolSchema(schema map[string]any, strict *bool) map[string]any {
	if schema == nil {
		return nil
	}
	normalized := make(map[string]any, len(schema)+2)
	for key, value := range schema {
		normalized[key] = value
	}
	if normalized["type"] == "object" {
		props, _ := normalized["properties"].(map[string]any)
		if props == nil {
			props = map[string]any{}
			normalized["properties"] = props
		}
		if _, ok := normalized["additionalProperties"].(map[string]any); ok {
			normalized["additionalProperties"] = false
		}
		if strict != nil && *strict {
			normalized["additionalProperties"] = false
			required := stringSetFromAnySlice(normalized["required"])
			for key := range props {
				required[key] = true
			}
			normalized["required"] = keysFromSet(required)
		}
	}
	return normalized
}

func stringSetFromAnySlice(value any) map[string]bool {
	set := map[string]bool{}
	items, ok := value.([]any)
	if !ok {
		return set
	}
	for _, item := range items {
		if text, ok := item.(string); ok && strings.TrimSpace(text) != "" {
			set[text] = true
		}
	}
	return set
}

func keysFromSet(set map[string]bool) []string {
	keys := make([]string, 0, len(set))
	for key := range set {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func decodeOpenAIResponseToolChoice(raw json.RawMessage) *ToolChoice {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}

	var value string
	if err := json.Unmarshal(raw, &value); err == nil {
		switch value {
		case "auto":
			return &ToolChoice{Type: ToolChoiceAuto}
		case "none":
			return &ToolChoice{Type: ToolChoiceNone}
		case "required":
			return &ToolChoice{Type: ToolChoiceRequired}
		}
	}

	var object struct {
		Type     string `json:"type"`
		Name     string `json:"name"`
		Function struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if err := json.Unmarshal(raw, &object); err != nil {
		return nil
	}
	if object.Type == "function" {
		name := object.Name
		if name == "" {
			name = object.Function.Name
		}
		return &ToolChoice{Type: ToolChoiceTool, ToolName: name}
	}
	return nil
}

func encodeOpenAIResponsesToolChoice(choice *ToolChoice) any {
	if choice == nil {
		return nil
	}
	switch choice.Type {
	case ToolChoiceAuto:
		return "auto"
	case ToolChoiceNone:
		return "none"
	case ToolChoiceRequired:
		return "required"
	case ToolChoiceTool:
		return map[string]any{"type": "function", "name": choice.ToolName}
	default:
		return nil
	}
}

func decodeOpenAIResponsesToolCall(item openAIResponsesOutputItem) (Part, error) {
	if item.Type == "custom_tool_call" {
		return Part{Type: PartToolCall, ToolCall: &ToolCallPart{ToolCallID: item.CallID, ToolName: item.Name, Input: item.Input, ProviderExecuted: true}}, nil
	}
	if item.Arguments == nil {
		return Part{Type: PartToolCall, ToolCall: &ToolCallPart{ToolCallID: item.CallID, ToolName: item.Name}}, nil
	}
	input, err := decodeOpenAIToolInput(item.Arguments.String())
	if err != nil {
		return Part{}, err
	}
	return Part{Type: PartToolCall, ToolCall: &ToolCallPart{ToolCallID: item.CallID, ToolName: item.Name, Input: input}}, nil
}

func encodeOpenAIResponsesToolCall(part *ToolCallPart, status string) (openAIResponsesOutputItem, error) {
	arguments, err := encodeOpenAIToolInput(part.Input)
	if err != nil {
		return openAIResponsesOutputItem{}, err
	}
	return openAIResponsesOutputItem{Type: "function_call", CallID: part.ToolCallID, Name: part.ToolName, Arguments: newOpenAIResponsesArgumentsString(arguments), Status: status}, nil
}

func decodeOpenAIResponsesToolResult(item openAIResponsesOutputItem) Part {
	return Part{Type: PartToolResult, ToolResult: &ToolResultPart{ToolCallID: item.CallID, Output: decodeOpenAIResponsesToolResultOutput(item.Output)}}
}

func decodeOpenAIResponsesToolResultOutput(output any) ToolResultOutput {
	if raw, ok := output.(json.RawMessage); ok {
		var text string
		if err := json.Unmarshal(raw, &text); err == nil {
			return ToolResultOutput{Type: ToolResultText, Text: text}
		}
		var input openAIResponsesToolOutputInput
		if err := json.Unmarshal(raw, &input); err == nil {
			return decodeOpenAIResponsesToolOutputInput(input)
		}
		var content []openAIResponsesContentPart
		if err := json.Unmarshal(raw, &content); err == nil {
			return ToolResultOutput{Type: ToolResultContent, Content: decodeOpenAIResponsesContentParts(content)}
		}
	}
	if text, ok := output.(string); ok {
		return ToolResultOutput{Type: ToolResultText, Text: text}
	}
	return ToolResultOutput{Type: ToolResultJSON, JSON: output}
}

func decodeOpenAIResponsesContentParts(parts []openAIResponsesContentPart) []Part {
	decoded := make([]Part, 0, len(parts))
	for _, part := range parts {
		if part.Type == "input_text" || part.Type == "output_text" || part.Type == "text" {
			decoded = append(decoded, Part{Type: PartText, Text: &TextPart{Text: part.Text}})
			continue
		}
		if part.Type == "input_image" {
			file := &FilePart{Type: FileImage, FileID: part.FileID, Detail: part.Detail}
			if part.ImageURL != "" {
				file = decodeFileURL(part.ImageURL, FileImage)
				file.FileID = part.FileID
				file.Detail = part.Detail
			}
			decoded = append(decoded, Part{Type: PartFile, File: file})
		}
	}
	return decoded
}

func encodeOpenAIResponsesToolResults(parts []Part) []openAIResponsesInputItem {
	items := make([]openAIResponsesInputItem, 0)
	for _, part := range parts {
		if part.Type != PartToolResult || part.ToolResult == nil {
			continue
		}
		items = append(items, openAIResponsesInputItem{Type: "function_call_output", CallID: part.ToolResult.ToolCallID, Output: encodeOpenAIResponsesToolOutput(part.ToolResult.Output)})
	}
	return items
}

func encodeOpenAIResponsesToolCalls(parts []Part) []openAIResponsesInputItem {
	items := make([]openAIResponsesInputItem, 0)
	for _, part := range parts {
		if part.Type != PartToolCall || part.ToolCall == nil {
			continue
		}
		item, err := encodeOpenAIResponsesToolCall(part.ToolCall, "completed")
		if err != nil {
			continue
		}
		items = append(items, openAIResponsesInputItemFromOutputItem(item))
	}
	return items
}

func openAIResponsesOutputItemFromInputItem(item openAIResponsesInputItem) openAIResponsesOutputItem {
	return openAIResponsesOutputItem{ID: item.ID, Type: item.Type, Role: item.Role, Status: item.Status, CallID: item.CallID, Name: item.Name, Arguments: item.Arguments, Input: item.Input, Output: item.Output, EncryptedContent: item.EncryptedContent}
}

func openAIResponsesInputItemFromOutputItem(item openAIResponsesOutputItem) openAIResponsesInputItem {
	return openAIResponsesInputItem{ID: item.ID, Type: item.Type, Role: item.Role, Status: item.Status, CallID: item.CallID, Name: item.Name, Arguments: item.Arguments, Input: item.Input, Output: item.Output, EncryptedContent: item.EncryptedContent}
}

func encodeOpenAIResponsesResponseToolCalls(parts []Part, reason FinishReason) []openAIResponsesOutputItem {
	items := make([]openAIResponsesOutputItem, 0)
	for _, part := range parts {
		if part.Type != PartToolCall || part.ToolCall == nil {
			continue
		}
		item, err := encodeOpenAIResponsesToolCall(part.ToolCall, encodeOpenAIResponsesStatus(reason))
		if err != nil {
			continue
		}
		items = append(items, item)
	}
	return items
}

func encodeOpenAIResponsesResponseToolResults(parts []Part, reason FinishReason) []openAIResponsesOutputItem {
	items := make([]openAIResponsesOutputItem, 0)
	for _, part := range parts {
		if part.Type != PartToolResult || part.ToolResult == nil {
			continue
		}
		items = append(items, openAIResponsesOutputItem{Type: "function_call_output", CallID: part.ToolResult.ToolCallID, Output: encodeOpenAIResponsesToolOutput(part.ToolResult.Output), Status: encodeOpenAIResponsesStatus(reason)})
	}
	return items
}

func encodeOpenAIResponsesToolOutput(output ToolResultOutput) any {
	if output.Type == ToolResultContent {
		items := make([]openAIResponsesContentPart, 0)
		for _, part := range output.Content {
			if part.Type == PartText && part.Text != nil {
				items = append(items, openAIResponsesContentPart{Type: "input_text", Text: part.Text.Text})
				continue
			}
			if part.Type == PartFile && part.File != nil && part.File.Type == FileImage {
				image := openAIResponsesContentPart{Type: "input_image", Detail: part.File.Detail}
				if image.Detail == "" {
					image.Detail = "auto"
				}
				if url := encodeFileURL(part.File); url != "" {
					image.ImageURL = url
				} else {
					image.FileID = part.File.FileID
				}
				if image.ImageURL != "" || image.FileID != "" {
					items = append(items, image)
				}
			}
		}
		if len(items) > 0 {
			return items
		}
		return ""
	}
	return fmt.Sprint(encodeOpenAIToolOutput(output))
}

type openAIResponsesToolOutputInput struct {
	Text  string                       `json:"text,omitempty"`
	Items []openAIResponsesContentPart `json:"items,omitempty"`
}

func decodeOpenAIResponsesToolOutputInput(input openAIResponsesToolOutputInput) ToolResultOutput {
	if len(input.Items) > 0 {
		return ToolResultOutput{Type: ToolResultContent, Content: decodeOpenAIResponsesContentParts(input.Items)}
	}
	return ToolResultOutput{Type: ToolResultText, Text: input.Text}
}

func decodeOpenAIResponsesStatus(status string) FinishReason {
	return decodeOpenAIResponsesFinishReason(status, nil)
}

func decodeOpenAIResponsesFinishReason(status string, details *openAIResponsesIncompleteDetails) FinishReason {
	switch status {
	case "completed":
		return FinishStop
	case "incomplete":
		if details == nil {
			return FinishLength
		}
		switch details.Reason {
		case "max_output_tokens":
			return FinishLength
		case "content_filter":
			return FinishContentFilter
		case "":
			return FinishLength
		default:
			return FinishOther
		}
	case "failed":
		return FinishError
	case "":
		return FinishUnknown
	default:
		return FinishOther
	}
}

func encodeOpenAIResponsesStatus(reason FinishReason) string {
	switch reason {
	case FinishError:
		return "failed"
	case FinishLength:
		return "incomplete"
	default:
		return "completed"
	}
}

func decodeOpenAIResponsesUsage(usage openAIResponsesUsage) Usage {
	decoded := Usage{InputTokens: usage.InputTokens, OutputTokens: usage.OutputTokens}
	if usage.InputTokensDetails != nil {
		decoded.CachedInputTokens = usage.InputTokensDetails.CachedTokens
	}
	if usage.OutputTokensDetails != nil {
		decoded.ReasoningTokens = usage.OutputTokensDetails.ReasoningTokens
	}
	return decoded
}

func encodeOpenAIResponsesUsage(usage Usage, billingUsage BillingUsage) openAIResponsesUsage {
	if hasBillingUsage(billingUsage) {
		inputTokens := billingUsage.InputTokens + billingUsage.CachedInputTokens
		outputTokens := billingUsage.OutputTokens
		cachedInputTokens := billingUsage.CachedInputTokens
		encoded := openAIResponsesUsage{InputTokens: &inputTokens, OutputTokens: &outputTokens, TotalTokens: calculateTotalTokens(&inputTokens, &outputTokens)}
		if cachedInputTokens > 0 {
			encoded.InputTokensDetails = &openAIResponsesInputTokensDetails{CachedTokens: &cachedInputTokens}
		}
		if usage.ReasoningTokens != nil {
			encoded.OutputTokensDetails = &openAIResponsesOutputTokensDetails{ReasoningTokens: usage.ReasoningTokens}
		}
		return encoded
	}
	encoded := openAIResponsesUsage{InputTokens: usage.InputTokens, OutputTokens: usage.OutputTokens, TotalTokens: calculateTotalTokens(usage.InputTokens, usage.OutputTokens)}
	if usage.CachedInputTokens != nil {
		encoded.InputTokensDetails = &openAIResponsesInputTokensDetails{CachedTokens: usage.CachedInputTokens}
	}
	if usage.ReasoningTokens != nil {
		encoded.OutputTokensDetails = &openAIResponsesOutputTokensDetails{ReasoningTokens: usage.ReasoningTokens}
	}
	return encoded
}

type openAIResponsesRequest struct {
	Model              string                `json:"model"`
	Instructions       string                `json:"instructions,omitempty"`
	Input              any                   `json:"input"`
	MaxOutputTokens    *int                  `json:"max_output_tokens,omitempty"`
	Temperature        *float64              `json:"temperature,omitempty"`
	TopP               *float64              `json:"top_p,omitempty"`
	Text               any                   `json:"text,omitempty"`
	Reasoning          any                   `json:"reasoning,omitempty"`
	Tools              []openAIResponsesTool `json:"tools,omitempty"`
	ToolChoice         any                   `json:"tool_choice,omitempty"`
	PreviousResponseID string                `json:"previous_response_id,omitempty"`
	Conversation       any                   `json:"conversation,omitempty"`
	Include            []string              `json:"include,omitempty"`
	ParallelToolCalls  *bool                 `json:"parallel_tool_calls,omitempty"`
	Stream             bool                  `json:"stream,omitempty"`
}

func (r *openAIResponsesRequest) UnmarshalJSON(raw []byte) error {
	type alias openAIResponsesRequest
	var decoded struct {
		alias
		Input        json.RawMessage `json:"input"`
		Text         json.RawMessage `json:"text"`
		Reasoning    json.RawMessage `json:"reasoning"`
		ToolChoice   json.RawMessage `json:"tool_choice"`
		Conversation json.RawMessage `json:"conversation"`
	}
	if err := json.Unmarshal(raw, &decoded); err != nil {
		return err
	}
	*r = openAIResponsesRequest(decoded.alias)
	r.Input = decoded.Input
	r.Text = decoded.Text
	r.Reasoning = decoded.Reasoning
	r.ToolChoice = decoded.ToolChoice
	r.Conversation = decoded.Conversation
	return nil
}

type openAIResponsesTextConfig struct {
	Format openAIResponsesTextFormat `json:"format"`
}

type openAIResponsesTextFormat struct {
	Type        string         `json:"type"`
	Name        string         `json:"name,omitempty"`
	Description string         `json:"description,omitempty"`
	Schema      map[string]any `json:"schema,omitempty"`
	Strict      *bool          `json:"strict,omitempty"`
}

type openAIResponsesArguments struct {
	value    string
	raw      json.RawMessage
	isString bool
}

func newOpenAIResponsesArgumentsString(value string) *openAIResponsesArguments {
	return &openAIResponsesArguments{value: value, isString: true}
}

func (a *openAIResponsesArguments) UnmarshalJSON(raw []byte) error {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || string(trimmed) == "null" {
		*a = openAIResponsesArguments{}
		return nil
	}
	var text string
	if err := json.Unmarshal(trimmed, &text); err == nil {
		*a = openAIResponsesArguments{value: text, isString: true}
		return nil
	}
	stored := append(json.RawMessage(nil), trimmed...)
	*a = openAIResponsesArguments{value: string(trimmed), raw: stored}
	return nil
}

func (a openAIResponsesArguments) MarshalJSON() ([]byte, error) {
	if !a.isString && len(a.raw) > 0 {
		return a.raw, nil
	}
	return json.Marshal(a.value)
}

func (a openAIResponsesArguments) IsZero() bool {
	return a.value == "" && len(a.raw) == 0
}

func (a openAIResponsesArguments) String() string {
	return a.value
}

type openAIResponsesInputItem struct {
	ID               string                       `json:"id,omitempty"`
	Type             string                       `json:"type,omitempty"`
	Role             string                       `json:"role,omitempty"`
	Content          any                          `json:"content,omitempty"`
	CallID           string                       `json:"call_id,omitempty"`
	Name             string                       `json:"name,omitempty"`
	Arguments        *openAIResponsesArguments    `json:"arguments,omitempty"`
	Input            any                          `json:"input,omitempty"`
	Output           any                          `json:"output,omitempty"`
	Status           string                       `json:"status,omitempty"`
	Summary          []openAIResponsesContentPart `json:"summary,omitempty"`
	EncryptedContent string                       `json:"encrypted_content,omitempty"`
}

func (m *openAIResponsesInputItem) UnmarshalJSON(raw []byte) error {
	var decoded struct {
		ID               string                    `json:"id"`
		Type             string                    `json:"type"`
		Role             string                    `json:"role"`
		Content          json.RawMessage           `json:"content"`
		CallID           string                    `json:"call_id"`
		Name             string                    `json:"name"`
		Arguments        *openAIResponsesArguments `json:"arguments"`
		Input            json.RawMessage           `json:"input"`
		Output           json.RawMessage           `json:"output"`
		Status           string                    `json:"status"`
		EncryptedContent string                    `json:"encrypted_content"`
	}
	if err := json.Unmarshal(raw, &decoded); err != nil {
		return err
	}
	m.ID = decoded.ID
	m.Type = decoded.Type
	m.Role = decoded.Role
	m.Content = decoded.Content
	m.CallID = decoded.CallID
	m.Name = decoded.Name
	m.Arguments = decoded.Arguments
	m.Input = decoded.Input
	m.Output = decoded.Output
	m.Status = decoded.Status
	m.EncryptedContent = decoded.EncryptedContent
	return nil
}

type openAIResponsesContentPart struct {
	Type        string                      `json:"type"`
	Text        string                      `json:"text,omitempty"`
	Refusal     string                      `json:"refusal,omitempty"`
	ImageURL    string                      `json:"image_url,omitempty"`
	FileID      string                      `json:"file_id,omitempty"`
	FileData    string                      `json:"file_data,omitempty"`
	FileURL     string                      `json:"file_url,omitempty"`
	Filename    string                      `json:"filename,omitempty"`
	Detail      string                      `json:"detail,omitempty"`
	Annotations []openAIResponsesAnnotation `json:"annotations,omitempty"`
}

type openAIResponsesTool struct {
	Type        string         `json:"type"`
	Name        string         `json:"name,omitempty"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
	Strict      *bool          `json:"strict,omitempty"`
	Raw         map[string]any `json:"-"`
}

func (t *openAIResponsesTool) UnmarshalJSON(raw []byte) error {
	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		return err
	}
	if value, ok := decoded["type"].(string); ok {
		t.Type = value
	}
	if value, ok := decoded["name"].(string); ok {
		t.Name = value
	}
	if value, ok := decoded["description"].(string); ok {
		t.Description = value
	}
	if value, ok := decoded["parameters"].(map[string]any); ok {
		t.Parameters = value
	}
	if value, ok := decoded["strict"].(bool); ok {
		t.Strict = &value
	}
	t.Raw = decoded
	return nil
}

func (t openAIResponsesTool) MarshalJSON() ([]byte, error) {
	if t.Type == "function" {
		type alias openAIResponsesTool
		return json.Marshal(struct {
			alias
			Raw any `json:"-"`
		}{alias: alias(t)})
	}
	return json.Marshal(t.Config())
}

func (t openAIResponsesTool) Config() map[string]any {
	config := make(map[string]any)
	for key, value := range t.Raw {
		config[key] = value
	}
	if t.Type != "" {
		config["type"] = t.Type
	}
	return config
}

func newOpenAIResponsesProviderTool(toolType string, config map[string]any) openAIResponsesTool {
	tool := openAIResponsesTool{Type: toolType, Raw: make(map[string]any)}
	for key, value := range config {
		tool.Raw[key] = value
	}
	tool.Raw["type"] = toolType
	return tool
}

type openAIResponsesResponse struct {
	ID                string                            `json:"id,omitempty"`
	Object            string                            `json:"object,omitempty"`
	Status            string                            `json:"status,omitempty"`
	IncompleteDetails *openAIResponsesIncompleteDetails `json:"incomplete_details,omitempty"`
	Model             string                            `json:"model,omitempty"`
	Output            []openAIResponsesOutputItem       `json:"output,omitempty"`
	OutputText        string                            `json:"output_text,omitempty"`
	Usage             openAIResponsesUsage              `json:"usage,omitempty"`
}

type openAIResponsesIncompleteDetails struct {
	Reason string `json:"reason,omitempty"`
}

type openAIResponsesOutputItem struct {
	ID               string                       `json:"id,omitempty"`
	Type             string                       `json:"type"`
	Role             string                       `json:"role,omitempty"`
	Status           string                       `json:"status,omitempty"`
	Content          []openAIResponsesContentPart `json:"content,omitempty"`
	Summary          []openAIResponsesContentPart `json:"summary,omitempty"`
	Text             string                       `json:"text,omitempty"`
	CallID           string                       `json:"call_id,omitempty"`
	Name             string                       `json:"name,omitempty"`
	Arguments        *openAIResponsesArguments    `json:"arguments,omitempty"`
	Input            any                          `json:"input,omitempty"`
	Output           any                          `json:"output,omitempty"`
	ImageURL         string                       `json:"image_url,omitempty"`
	Result           string                       `json:"result,omitempty"`
	OutputFormat     string                       `json:"output_format,omitempty"`
	EncryptedContent string                       `json:"encrypted_content,omitempty"`
}

type openAIResponsesUsage struct {
	InputTokens         *int                                `json:"input_tokens,omitempty"`
	OutputTokens        *int                                `json:"output_tokens,omitempty"`
	TotalTokens         *int                                `json:"total_tokens,omitempty"`
	InputTokensDetails  *openAIResponsesInputTokensDetails  `json:"input_tokens_details,omitempty"`
	OutputTokensDetails *openAIResponsesOutputTokensDetails `json:"output_tokens_details,omitempty"`
}

type openAIResponsesInputTokensDetails struct {
	CachedTokens *int `json:"cached_tokens,omitempty"`
}

type openAIResponsesOutputTokensDetails struct {
	ReasoningTokens *int `json:"reasoning_tokens,omitempty"`
}

type openAIResponsesStreamEvent struct {
	Type         string                            `json:"type"`
	Response     *openAIResponsesStreamResponse    `json:"response,omitempty"`
	Item         *openAIResponsesStreamItem        `json:"item,omitempty"`
	ItemID       string                            `json:"item_id,omitempty"`
	ContentPart  *openAIResponsesStreamContentPart `json:"content_part,omitempty"`
	Index        int                               `json:"index,omitempty"`
	OutputIndex  *int                              `json:"output_index,omitempty"`
	ContentIndex *int                              `json:"content_index,omitempty"`
	Delta        string                            `json:"delta,omitempty"`
	Arguments    string                            `json:"arguments,omitempty"`
	Usage        *openAIResponsesUsage             `json:"usage,omitempty"`
	Error        any                               `json:"error,omitempty"`
	Output       []openAIResponsesStreamItem       `json:"output,omitempty"`
}

type openAIResponsesStreamResponse struct {
	ID                string                            `json:"id,omitempty"`
	Object            string                            `json:"object,omitempty"`
	Status            string                            `json:"status,omitempty"`
	Error             any                               `json:"error,omitempty"`
	IncompleteDetails *openAIResponsesIncompleteDetails `json:"incomplete_details,omitempty"`
	Model             string                            `json:"model,omitempty"`
	Usage             *openAIResponsesUsage             `json:"usage,omitempty"`
}

type openAIResponsesStreamItem struct {
	ID               string                       `json:"id,omitempty"`
	Type             string                       `json:"type,omitempty"`
	Role             string                       `json:"role,omitempty"`
	Status           string                       `json:"status,omitempty"`
	Name             string                       `json:"name,omitempty"`
	CallID           string                       `json:"call_id,omitempty"`
	Arguments        *openAIResponsesArguments    `json:"arguments,omitempty"`
	Input            any                          `json:"input,omitempty"`
	Content          []openAIResponsesContentPart `json:"content,omitempty"`
	Summary          []openAIResponsesContentPart `json:"summary,omitempty"`
	EncryptedContent string                       `json:"encrypted_content,omitempty"`
}

type openAIResponsesStreamContentPart struct {
	Type        string                      `json:"type,omitempty"`
	Text        string                      `json:"text,omitempty"`
	Annotations []openAIResponsesAnnotation `json:"annotations,omitempty"`
}

type openAIResponsesAnnotation struct {
	Type     string `json:"type,omitempty"`
	Index    int    `json:"index,omitempty"`
	FileID   string `json:"file_id,omitempty"`
	Filename string `json:"filename,omitempty"`
}

type openAIResponsesStreamDecoder struct {
	started     bool
	activeTools map[string]openAIResponsesStreamToolState
	endedTools  map[string]bool
	hasToolCall bool
}

type openAIResponsesStreamToolState struct {
	CallID string
	Name   string
	Custom bool
}

func (d *openAIResponsesStreamDecoder) Decode(event RawStreamEvent) ([]StreamPart, error) {
	if len(bytes.TrimSpace(event.Data)) == 0 {
		return nil, nil
	}

	var raw openAIResponsesStreamEvent
	if err := json.Unmarshal(event.Data, &raw); err != nil {
		return []StreamPart{{Type: StreamRaw, RawValue: string(event.Data)}}, nil
	}

	switch raw.Type {
	case "response.created", "response.in_progress":
		if !d.started {
			d.started = true
			part := StreamPart{Type: StreamStart}
			if raw.Response != nil {
				part.ID = raw.Response.ID
				part.ProviderMetadata = map[string]any{"model": raw.Response.Model, "status": raw.Response.Status}
				if raw.Response.Usage != nil {
					part.Usage = decodeOpenAIResponsesUsage(*raw.Response.Usage)
				}
			}
			return []StreamPart{part}, nil
		}
		return nil, nil
	case "response.output_item.added":
		if raw.Item == nil {
			return nil, nil
		}
		switch raw.Item.Type {
		case "reasoning":
			return []StreamPart{{Type: StreamReasoningStart, ID: raw.Item.ID, ProviderMetadata: map[string]any{"status": raw.Item.Status}}}, nil
		case "message":
			return []StreamPart{{Type: StreamTextStart, ID: raw.Item.ID, ProviderMetadata: map[string]any{"status": raw.Item.Status}}}, nil
		case "function_call":
			d.setActiveTool(raw.Item.ID, raw.Item.CallID, raw.Item.Name, false)
			d.hasToolCall = true
			return []StreamPart{{Type: StreamToolInputStart, ID: raw.Item.ID, ToolCallID: raw.Item.CallID, ToolName: raw.Item.Name}}, nil
		case "custom_tool_call":
			d.setActiveTool(raw.Item.ID, raw.Item.CallID, raw.Item.Name, true)
			d.hasToolCall = true
			return []StreamPart{{Type: StreamToolInputStart, ID: raw.Item.ID, ToolCallID: raw.Item.CallID, ToolName: raw.Item.Name, ProviderMetadata: map[string]any{"custom_tool_call": true}}}, nil
		default:
			return []StreamPart{{Type: StreamRaw, RawValue: raw}}, nil
		}
	case "response.content_part.added":
		if raw.ContentPart == nil {
			return nil, nil
		}
		switch raw.ContentPart.Type {
		case "reasoning_text", "summary_text":
			return []StreamPart{{Type: StreamReasoningStart}}, nil
		case "output_text", "text":
			return []StreamPart{{Type: StreamTextStart}}, nil
		case "refusal":
			return []StreamPart{{Type: StreamTextStart, ProviderMetadata: map[string]any{"refusal": true}}}, nil
		default:
			return nil, nil
		}
	case "response.output_text.delta":
		return []StreamPart{{Type: StreamTextDelta, Delta: raw.Delta}}, nil
	case "response.reasoning_summary_text.delta", "response.reasoning.delta", "response.reasoning_text.delta":
		return []StreamPart{{Type: StreamReasoningDelta, Delta: raw.Delta}}, nil
	case "response.function_call_arguments.delta":
		delta := raw.Delta
		if delta == "" {
			delta = raw.Arguments
		}
		state := d.activeTool(raw.ItemID)
		return []StreamPart{{Type: StreamToolInputDelta, ID: raw.ItemID, ToolCallID: state.CallID, ToolName: state.Name, Delta: delta}}, nil
	case "response.custom_tool_call_input.delta":
		state := d.activeTool(raw.ItemID)
		return []StreamPart{{Type: StreamToolInputDelta, ID: raw.ItemID, ToolCallID: state.CallID, ToolName: state.Name, Delta: raw.Delta, ProviderMetadata: map[string]any{"custom_tool_call": true}}}, nil
	case "response.output_text.done":
		return []StreamPart{{Type: StreamTextEnd}}, nil
	case "response.reasoning_summary_text.done", "response.reasoning.done", "response.reasoning_text.done":
		return []StreamPart{{Type: StreamReasoningEnd}}, nil
	case "response.function_call_arguments.done":
		state := d.activeTool(raw.ItemID)
		d.clearActiveTool(raw.ItemID)
		d.markToolEnded(raw.ItemID)
		return []StreamPart{{Type: StreamToolInputEnd, ID: raw.ItemID, ToolCallID: state.CallID, ToolName: state.Name}}, nil
	case "response.custom_tool_call_input.done":
		state := d.activeTool(raw.ItemID)
		d.clearActiveTool(raw.ItemID)
		d.markToolEnded(raw.ItemID)
		return []StreamPart{{Type: StreamToolInputEnd, ID: raw.ItemID, ToolCallID: state.CallID, ToolName: state.Name, ProviderMetadata: map[string]any{"custom_tool_call": true}}}, nil
	case "response.content_part.done":
		return nil, nil
	case "response.output_item.done":
		if raw.Item == nil {
			return nil, nil
		}
		switch raw.Item.Type {
		case "reasoning":
			return []StreamPart{{Type: StreamReasoningEnd, ID: raw.Item.ID}}, nil
		case "message":
			return []StreamPart{{Type: StreamTextEnd, ID: raw.Item.ID}}, nil
		case "function_call":
			if d.toolEnded(raw.Item.ID) {
				return nil, nil
			}
			d.hasToolCall = true
			d.markToolEnded(raw.Item.ID)
			return []StreamPart{{Type: StreamToolInputEnd, ID: raw.Item.ID, ToolCallID: raw.Item.CallID, ToolName: raw.Item.Name}}, nil
		case "custom_tool_call":
			if d.toolEnded(raw.Item.ID) {
				return nil, nil
			}
			d.hasToolCall = true
			d.markToolEnded(raw.Item.ID)
			return []StreamPart{{Type: StreamToolInputEnd, ID: raw.Item.ID, ToolCallID: raw.Item.CallID, ToolName: raw.Item.Name, ProviderMetadata: map[string]any{"custom_tool_call": true}}}, nil
		default:
			return nil, nil
		}
	case "response.completed":
		parts := make([]StreamPart, 0, 2)
		finish := StreamPart{Type: StreamFinish}
		if raw.Response != nil {
			finish.ID = raw.Response.ID
			finish.FinishReason = decodeOpenAIResponsesFinishReason(raw.Response.Status, raw.Response.IncompleteDetails)
			if finish.FinishReason == FinishStop && d.hasToolCall {
				finish.FinishReason = FinishToolCalls
			}
			if raw.Response.Usage != nil {
				finish.Usage = decodeOpenAIResponsesUsage(*raw.Response.Usage)
			}
		}
		parts = append(parts, finish)
		return parts, nil
	case "response.incomplete":
		parts := make([]StreamPart, 0, 2)
		finish := StreamPart{Type: StreamFinish, FinishReason: FinishOther}
		if raw.Response != nil {
			finish.ID = raw.Response.ID
			finish.FinishReason = decodeOpenAIResponsesFinishReason(raw.Response.Status, raw.Response.IncompleteDetails)
			if raw.Response.Usage != nil {
				finish.Usage = decodeOpenAIResponsesUsage(*raw.Response.Usage)
			}
		}
		parts = append(parts, finish)
		return parts, nil
	case "response.failed":
		return []StreamPart{{Type: StreamError, Error: raw.Error}}, nil
	case "error":
		return []StreamPart{{Type: StreamError, Error: raw.Error}}, nil
	default:
		return []StreamPart{{Type: StreamRaw, RawValue: raw}}, nil
	}
}

func (d *openAIResponsesStreamDecoder) setActiveTool(itemID string, callID string, name string, custom bool) {
	if itemID == "" {
		return
	}
	if d.activeTools == nil {
		d.activeTools = make(map[string]openAIResponsesStreamToolState)
	}
	d.activeTools[itemID] = openAIResponsesStreamToolState{CallID: callID, Name: name, Custom: custom}
}

func (d *openAIResponsesStreamDecoder) markToolEnded(itemID string) {
	if itemID == "" {
		return
	}
	if d.endedTools == nil {
		d.endedTools = make(map[string]bool)
	}
	d.endedTools[itemID] = true
}

func (d *openAIResponsesStreamDecoder) toolEnded(itemID string) bool {
	if d.endedTools == nil || itemID == "" {
		return false
	}
	return d.endedTools[itemID]
}

func (d *openAIResponsesStreamDecoder) activeTool(itemID string) openAIResponsesStreamToolState {
	if d.activeTools == nil || itemID == "" {
		return openAIResponsesStreamToolState{}
	}
	return d.activeTools[itemID]
}

func (d *openAIResponsesStreamDecoder) clearActiveTool(itemID string) {
	if d.activeTools == nil || itemID == "" {
		return
	}
	delete(d.activeTools, itemID)
}

func (d *openAIResponsesStreamDecoder) Close() ([]StreamPart, error) {
	return nil, nil
}

type openAIResponsesStreamEncoder struct {
	model        string
	started      bool
	finished     bool
	responseID   string
	itemCounter  int
	currentMsgID string
	msgText      string
	currentRsID  string
	rsSummary    string
	rsEncrypted  string
	currentFcID  string
	fcName       string
	fcCallID     string
	fcArguments  string
	outputItems  []openAIResponsesStreamItem
}

func (e *openAIResponsesStreamEncoder) Encode(part StreamPart) ([]RawStreamEvent, error) {
	switch part.Type {
	case StreamStart:
		e.started = true
		e.responseID = part.ID
		resp := openAIResponsesStreamResponse{ID: part.ID, Object: "response", Status: "in_progress", Model: e.model}
		if part.Usage.InputTokens != nil || part.Usage.OutputTokens != nil {
			u := encodeOpenAIResponsesUsage(part.Usage, billingUsageForProtocol(ProtocolOpenAIResponses, part.Usage))
			resp.Usage = &u
		}
		created, err := singleOpenAIResponsesStreamEvent("response.created", openAIResponsesStreamEvent{Type: "response.created", Response: &resp})
		if err != nil {
			return nil, err
		}
		inProgress, err := singleOpenAIResponsesStreamEvent("response.in_progress", openAIResponsesStreamEvent{Type: "response.in_progress", Response: &resp})
		if err != nil {
			return nil, err
		}
		return append(created, inProgress...), nil
	case StreamTextStart:
		itemID := e.nextItemID("msg")
		e.currentMsgID = itemID
		e.msgText = ""
		msgItem := openAIResponsesStreamItem{ID: itemID, Type: "message", Role: string(RoleAssistant), Status: "in_progress", Content: []openAIResponsesContentPart{{Type: "output_text", Text: ""}}}
		outputIndex := len(e.outputItems)
		added, err := singleOpenAIResponsesStreamEvent("response.output_item.added", openAIResponsesStreamEvent{Type: "response.output_item.added", OutputIndex: intPtr(outputIndex), Item: &msgItem})
		if err != nil {
			return nil, err
		}
		partAdded, err := singleOpenAIResponsesStreamEvent("response.content_part.added", openAIResponsesStreamEvent{Type: "response.content_part.added", ItemID: itemID, OutputIndex: intPtr(outputIndex), ContentIndex: intPtr(0), ContentPart: &openAIResponsesStreamContentPart{Type: "output_text", Text: ""}})
		if err != nil {
			return nil, err
		}
		return append(added, partAdded...), nil
	case StreamTextDelta:
		events := make([]RawStreamEvent, 0, 3)
		if e.currentMsgID == "" {
			startEvents, err := e.Encode(StreamPart{Type: StreamTextStart, ID: part.ID})
			if err != nil {
				return nil, err
			}
			events = append(events, startEvents...)
		}
		e.msgText += part.Delta
		deltaEvents, err := singleOpenAIResponsesStreamEvent("response.output_text.delta", openAIResponsesStreamEvent{Type: "response.output_text.delta", ItemID: e.currentMsgID, OutputIndex: intPtr(len(e.outputItems)), ContentIndex: intPtr(0), Delta: part.Delta})
		if err != nil {
			return nil, err
		}
		return append(events, deltaEvents...), nil
	case StreamTextEnd:
		outputIndex := len(e.outputItems)
		done, err := singleOpenAIResponsesStreamEvent("response.output_text.done", openAIResponsesStreamEvent{Type: "response.output_text.done", ItemID: e.currentMsgID, OutputIndex: intPtr(outputIndex), ContentIndex: intPtr(0)})
		if err != nil {
			return nil, err
		}
		partDone, err := singleOpenAIResponsesStreamEvent("response.content_part.done", openAIResponsesStreamEvent{Type: "response.content_part.done", ItemID: e.currentMsgID, OutputIndex: intPtr(outputIndex), ContentIndex: intPtr(0), ContentPart: &openAIResponsesStreamContentPart{Type: "output_text", Text: e.msgText, Annotations: []openAIResponsesAnnotation{}}})
		if err != nil {
			return nil, err
		}
		itemDone, err := singleOpenAIResponsesStreamEvent("response.output_item.done", openAIResponsesStreamEvent{Type: "response.output_item.done", OutputIndex: intPtr(outputIndex), Item: &openAIResponsesStreamItem{ID: e.currentMsgID, Type: "message", Role: string(RoleAssistant), Status: "completed", Content: []openAIResponsesContentPart{{Type: "output_text", Text: e.msgText, Annotations: []openAIResponsesAnnotation{}}}}})
		if err != nil {
			return nil, err
		}
		e.outputItems = append(e.outputItems, openAIResponsesStreamItem{ID: e.currentMsgID, Type: "message", Role: string(RoleAssistant), Status: "completed", Content: []openAIResponsesContentPart{{Type: "output_text", Text: e.msgText, Annotations: []openAIResponsesAnnotation{}}}})
		e.currentMsgID = ""
		e.msgText = ""
		return append(append(done, partDone...), itemDone...), nil
	case StreamReasoningStart:
		itemID := e.nextItemID("rs")
		e.currentRsID = itemID
		e.rsSummary = ""
		e.rsEncrypted = ""
		rsItem := openAIResponsesStreamItem{ID: itemID, Type: "reasoning", Status: "in_progress"}
		return singleOpenAIResponsesStreamEvent("response.output_item.added", openAIResponsesStreamEvent{Type: "response.output_item.added", OutputIndex: intPtr(len(e.outputItems)), Item: &rsItem})
	case StreamReasoningDelta:
		if signature, ok := part.ProviderMetadata["signature"].(string); ok && signature != "" {
			e.rsEncrypted += signature
			return nil, nil
		}
		e.rsSummary += part.Delta
		return singleOpenAIResponsesStreamEvent("response.reasoning_summary_text.delta", openAIResponsesStreamEvent{Type: "response.reasoning_summary_text.delta", ItemID: e.currentRsID, OutputIndex: intPtr(len(e.outputItems)), Delta: part.Delta})
	case StreamReasoningEnd:
		item := openAIResponsesStreamItem{ID: e.currentRsID, Type: "reasoning", Status: "completed", EncryptedContent: e.rsEncrypted}
		if e.rsSummary != "" {
			item.Summary = []openAIResponsesContentPart{{Type: "summary_text", Text: e.rsSummary}}
		}
		itemDone, err := singleOpenAIResponsesStreamEvent("response.output_item.done", openAIResponsesStreamEvent{Type: "response.output_item.done", OutputIndex: intPtr(len(e.outputItems)), Item: &item})
		if err != nil {
			return nil, err
		}
		e.outputItems = append(e.outputItems, item)
		e.currentRsID = ""
		e.rsSummary = ""
		e.rsEncrypted = ""
		return itemDone, nil
	case StreamToolInputStart:
		itemID := e.nextItemID("fc")
		e.currentFcID = itemID
		e.fcName = part.ToolName
		e.fcCallID = part.ToolCallID
		e.fcArguments = ""
		if custom, ok := part.ProviderMetadata["custom_tool_call"].(bool); ok && custom {
			fcItem := openAIResponsesStreamItem{ID: itemID, Type: "custom_tool_call", Status: "in_progress", Name: part.ToolName, CallID: part.ToolCallID, Input: ""}
			return singleOpenAIResponsesStreamEvent("response.output_item.added", openAIResponsesStreamEvent{Type: "response.output_item.added", OutputIndex: intPtr(len(e.outputItems)), Item: &fcItem})
		}
		fcItem := openAIResponsesStreamItem{ID: itemID, Type: "function_call", Status: "in_progress", Name: part.ToolName, CallID: part.ToolCallID}
		return singleOpenAIResponsesStreamEvent("response.output_item.added", openAIResponsesStreamEvent{Type: "response.output_item.added", OutputIndex: intPtr(len(e.outputItems)), Item: &fcItem})
	case StreamToolInputDelta:
		if e.currentFcID == "" {
			e.currentFcID = e.nextItemID("fc")
		}
		e.fcArguments += part.Delta
		if custom, ok := part.ProviderMetadata["custom_tool_call"].(bool); ok && custom {
			return singleOpenAIResponsesStreamEvent("response.custom_tool_call_input.delta", openAIResponsesStreamEvent{Type: "response.custom_tool_call_input.delta", ItemID: e.currentFcID, OutputIndex: intPtr(len(e.outputItems)), Delta: part.Delta})
		}
		return singleOpenAIResponsesStreamEvent("response.function_call_arguments.delta", openAIResponsesStreamEvent{Type: "response.function_call_arguments.delta", ItemID: e.currentFcID, OutputIndex: intPtr(len(e.outputItems)), Delta: part.Delta})
	case StreamToolInputEnd:
		itemID := e.currentFcID
		if e.currentFcID != "" {
			custom, _ := part.ProviderMetadata["custom_tool_call"].(bool)
			itemType := "function_call"
			item := openAIResponsesStreamItem{ID: e.currentFcID, Type: itemType, Status: "completed", Name: e.fcName, CallID: e.fcCallID, Arguments: newOpenAIResponsesArgumentsString(e.fcArguments)}
			if custom {
				item.Type = "custom_tool_call"
				item.Arguments = nil
				item.Input = e.fcArguments
			}
			e.outputItems = append(e.outputItems, item)
			e.currentFcID = ""
		}
		outputIndex := len(e.outputItems) - 1
		if outputIndex < 0 {
			outputIndex = 0
		}
		if custom, ok := part.ProviderMetadata["custom_tool_call"].(bool); ok && custom {
			return singleOpenAIResponsesStreamEvent("response.custom_tool_call_input.done", openAIResponsesStreamEvent{Type: "response.custom_tool_call_input.done", ItemID: itemID, OutputIndex: intPtr(outputIndex)})
		}
		return singleOpenAIResponsesStreamEvent("response.function_call_arguments.done", openAIResponsesStreamEvent{Type: "response.function_call_arguments.done", ItemID: itemID, OutputIndex: intPtr(outputIndex)})
	case StreamToolCall:
		return e.encodeToolCall(part)
	case StreamFinish:
		e.finished = true
		return e.encodeFinish(part)
	case StreamResponseMetadata:
		return nil, nil
	case StreamError:
		return e.encodeStreamError(part)
	case StreamRaw:
		return singleOpenAIResponsesStreamEvent("raw", openAIResponsesStreamEvent{Type: "raw", Delta: fmt.Sprint(part.RawValue)})
	default:
		return nil, nil
	}
}

func (e *openAIResponsesStreamEncoder) Close() ([]RawStreamEvent, error) {
	if e.finished {
		return nil, nil
	}
	return e.encodeFinish(StreamPart{Type: StreamFinish, FinishReason: FinishStop})
}

func (e *openAIResponsesStreamEncoder) EncodeError(err error) []RawStreamEvent {
	message := "unknown error"
	if err != nil {
		message = err.Error()
	}
	part := StreamPart{Type: StreamError, Error: map[string]any{"message": message, "type": "protocol_bridge_error"}}
	events, _ := e.encodeStreamError(part)
	return events
}

func (e *openAIResponsesStreamEncoder) encodeToolCall(part StreamPart) ([]RawStreamEvent, error) {
	toolID := part.ToolCallID
	if toolID == "" {
		toolID = part.ID
	}
	name := part.ToolName
	input, err := encodeOpenAIToolInput(part.Input)
	if err != nil {
		return nil, err
	}
	itemID := e.nextItemID("fc")
	var events []RawStreamEvent
	outputIndex := len(e.outputItems)
	added, err := singleOpenAIResponsesStreamEvent("response.output_item.added", openAIResponsesStreamEvent{Type: "response.output_item.added", OutputIndex: intPtr(outputIndex), Item: &openAIResponsesStreamItem{ID: itemID, Type: "function_call", Status: "in_progress", Name: name, CallID: toolID}})
	if err != nil {
		return nil, err
	}
	events = append(events, added...)
	delta, err := singleOpenAIResponsesStreamEvent("response.function_call_arguments.delta", openAIResponsesStreamEvent{Type: "response.function_call_arguments.delta", ItemID: itemID, OutputIndex: intPtr(outputIndex), Delta: input})
	if err != nil {
		return nil, err
	}
	events = append(events, delta...)
	done, err := singleOpenAIResponsesStreamEvent("response.function_call_arguments.done", openAIResponsesStreamEvent{Type: "response.function_call_arguments.done", ItemID: itemID, OutputIndex: intPtr(outputIndex)})
	if err != nil {
		return nil, err
	}
	events = append(events, done...)
	itemDone, err := singleOpenAIResponsesStreamEvent("response.output_item.done", openAIResponsesStreamEvent{Type: "response.output_item.done", OutputIndex: intPtr(outputIndex), Item: &openAIResponsesStreamItem{ID: itemID, Type: "function_call", Status: "completed", Name: name, CallID: toolID, Arguments: newOpenAIResponsesArgumentsString(input)}})
	if err != nil {
		return nil, err
	}
	events = append(events, itemDone...)
	e.outputItems = append(e.outputItems, openAIResponsesStreamItem{ID: itemID, Type: "function_call", Status: "completed", Name: name, CallID: toolID, Arguments: newOpenAIResponsesArgumentsString(input)})
	return events, nil
}

func (e *openAIResponsesStreamEncoder) encodeFinish(part StreamPart) ([]RawStreamEvent, error) {
	status := "completed"
	eventType := "response.completed"
	var incompleteDetails *openAIResponsesIncompleteDetails
	if part.FinishReason == FinishLength {
		status = "incomplete"
		eventType = "response.incomplete"
		incompleteDetails = &openAIResponsesIncompleteDetails{Reason: "max_output_tokens"}
	} else if part.FinishReason == FinishContentFilter {
		status = "incomplete"
		eventType = "response.incomplete"
		incompleteDetails = &openAIResponsesIncompleteDetails{Reason: "content_filter"}
	} else if part.FinishReason == FinishError {
		status = "failed"
		eventType = "response.failed"
	}
	resp := openAIResponsesStreamResponse{ID: e.responseID, Object: "response", Status: status, Error: part.Error, IncompleteDetails: incompleteDetails, Model: e.model}
	if part.Usage.InputTokens != nil || part.Usage.OutputTokens != nil {
		u := encodeOpenAIResponsesUsage(part.Usage, billingUsageForProtocol(ProtocolOpenAIResponses, part.Usage))
		resp.Usage = &u
	}
	outputItems := make([]openAIResponsesStreamItem, len(e.outputItems))
	copy(outputItems, e.outputItems)
	return singleOpenAIResponsesStreamEvent(eventType, openAIResponsesStreamEvent{Type: eventType, Response: &resp, Output: outputItems})
}

func (e *openAIResponsesStreamEncoder) encodeStreamError(part StreamPart) ([]RawStreamEvent, error) {
	return singleOpenAIResponsesStreamEvent("error", openAIResponsesStreamEvent{Type: "error", Error: part.Error})
}

func (e *openAIResponsesStreamEncoder) nextItemID(prefix string) string {
	e.itemCounter++
	return fmt.Sprintf("%s_%d", prefix, e.itemCounter)
}

func singleOpenAIResponsesStreamEvent(event string, payload openAIResponsesStreamEvent) ([]RawStreamEvent, error) {
	raw, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	return []RawStreamEvent{{Event: event, Data: raw}}, nil
}
