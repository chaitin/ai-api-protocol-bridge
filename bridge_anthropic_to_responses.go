package protocolbridge

import (
	"encoding/json"
	"fmt"
	"strings"
)

func (b anthropicToOpenAIResponsesBridge) InboundProtocol() Protocol {
	return ProtocolAnthropicMessages
}

func (b anthropicToOpenAIResponsesBridge) UpstreamProtocol() Protocol {
	return ProtocolOpenAIResponses
}

func (b anthropicToOpenAIResponsesBridge) EncodeUpstreamRequest(req *LLMRequest, opts EncodeRequestOptions) ([]byte, error) {
	if req == nil {
		return nil, fmt.Errorf("encode anthropic to openai responses request: nil request")
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
		Include:           append([]string(nil), req.Include...),
		ParallelToolCalls: req.ParallelToolCalls,
		Stream:            req.Stream,
	}
	if len(request.Tools) == 0 {
		request.ParallelToolCalls = nil
	}
	encodeOpenAIResponsesState(&request, req.State)

	input := make([]openAIResponsesInputItem, 0)
	for _, message := range req.Prompt {
		switch message.Role {
		case RoleSystem, RoleDeveloper:
			request.Instructions = appendInstructionsText(request.Instructions, joinTextParts(message.Parts))
		case RoleTool:
			input = append(input, encodeOpenAIResponsesToolResults(message.Parts)...)
		default:
			input = append(input, anthropicBridgeInputItemsForMessage(message)...)
		}
	}
	request.Input = input

	return json.Marshal(request)
}

func (b anthropicToOpenAIResponsesBridge) DecodeUpstreamResponse(raw []byte) (*LLMResponse, error) {
	resp, err := b.adapter.DecodeResponse(raw)
	if err != nil {
		return nil, err
	}
	resp.Protocol = ProtocolOpenAIResponses
	return resp, nil
}

func (b anthropicToOpenAIResponsesBridge) NewStreamDecoder(opts StreamDecodeOptions) (StreamDecoder, error) {
	return (&OpenAIResponsesAdapter{}).NewStreamDecoder(opts)
}

func (b anthropicToOpenAIResponsesBridge) NewStreamEncoder(opts StreamEncodeOptions) (StreamEncoder, error) {
	return &openAIResponsesToAnthropicStreamEncoder{ant: anthropicStreamEncoder{model: opts.Model}}, nil
}

func anthropicBridgeInputItemsForMessage(message Message) []openAIResponsesInputItem {
	items := make([]openAIResponsesInputItem, 0, 2)
	content := make([]openAIResponsesContentPart, 0, len(message.Parts))

	for _, part := range message.Parts {
		switch part.Type {
		case PartReasoning:
			if part.Reasoning == nil {
				continue
			}
			encrypted := strings.TrimSpace(part.Reasoning.Redacted)
			if encrypted == "" {
				encrypted = strings.TrimSpace(part.Reasoning.Signature)
			}
			if encrypted != "" {
				summary := make([]openAIResponsesContentPart, 0, 1)
				if strings.TrimSpace(part.Reasoning.Text) != "" {
					summary = append(summary, openAIResponsesContentPart{Type: "summary_text", Text: part.Reasoning.Text})
				}
				items = append(items, openAIResponsesInputItem{Type: "reasoning", Summary: summary, EncryptedContent: encrypted})
			}
		case PartText:
			if part.Text == nil {
				continue
			}
			partType := "input_text"
			if message.Role == RoleAssistant {
				partType = "output_text"
			}
			content = append(content, openAIResponsesContentPart{Type: partType, Text: part.Text.Text})
		case PartFile:
			if part.File == nil || message.Role == RoleAssistant {
				continue
			}
			switch part.File.Type {
			case FileImage:
				url := encodeFileURL(part.File)
				if url == "" {
					continue
				}
				detail := part.File.Detail
				if detail == "" {
					detail = "auto"
				}
				content = append(content, openAIResponsesContentPart{Type: "input_image", ImageURL: url, Detail: detail})
			case FileDocument:
				file := openAIResponsesContentPart{Type: "input_file", FileData: part.File.Data, FileURL: part.File.URL, FileID: part.File.FileID, Filename: part.File.Filename, Detail: part.File.Detail}
				if file.FileData == "" && file.FileURL == "" && file.FileID == "" {
					continue
				}
				content = append(content, file)
			}
		case PartToolCall:
			if part.ToolCall == nil {
				continue
			}
			arguments, err := encodeOpenAIToolInput(part.ToolCall.Input)
			if err != nil {
				continue
			}
			items = append(items, openAIResponsesInputItem{Type: "function_call", CallID: part.ToolCall.ToolCallID, Name: part.ToolCall.ToolName, Arguments: newOpenAIResponsesArgumentsString(arguments), Status: "completed"})
		case PartToolResult:
			if part.ToolResult == nil {
				continue
			}
			items = append(items, openAIResponsesInputItem{Type: "function_call_output", CallID: part.ToolResult.ToolCallID, Output: encodeOpenAIResponsesToolOutput(part.ToolResult.Output), Status: "completed"})
		}
	}

	if len(content) > 0 {
		messageItem := openAIResponsesInputItem{Type: "message", Role: string(message.Role), Content: content}
		if message.Role == RoleAssistant {
			messageItem.Status = "completed"
		}
		insertAt := len(items)
		for i, item := range items {
			if item.Type == "function_call" {
				insertAt = i
				break
			}
		}
		items = append(items, openAIResponsesInputItem{})
		copy(items[insertAt+1:], items[insertAt:])
		items[insertAt] = messageItem
	}

	return items
}

type openAIResponsesToAnthropicStreamEncoder struct {
	ant anthropicStreamEncoder
}

func (e *openAIResponsesToAnthropicStreamEncoder) Encode(part StreamPart) ([]RawStreamEvent, error) {
	if part.Type == StreamStart || part.Type == StreamFinish {
		part.Usage = responsesUsageToAnthropicUsage(part.Usage)
	}
	if part.Type == StreamReasoningDelta {
		if signature, ok := part.ProviderMetadata["signature"].(string); ok && signature != "" && part.Delta == "" {
			return e.ant.Encode(StreamPart{Type: StreamReasoningDelta, ID: part.ID, ProviderMetadata: map[string]any{"signature": signature}})
		}
	}
	return e.ant.Encode(part)
}

func (e *openAIResponsesToAnthropicStreamEncoder) Close() ([]RawStreamEvent, error) {
	return e.ant.Close()
}

func (e *openAIResponsesToAnthropicStreamEncoder) EncodeError(err error) []RawStreamEvent {
	return e.ant.EncodeError(err)
}
