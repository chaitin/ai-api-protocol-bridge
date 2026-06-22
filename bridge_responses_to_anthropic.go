package protocolbridge

import (
	"encoding/json"
	"fmt"
)

func (b openAIResponsesToAnthropicBridge) InboundProtocol() Protocol {
	return ProtocolOpenAIResponses
}

func (b openAIResponsesToAnthropicBridge) UpstreamProtocol() Protocol {
	return ProtocolAnthropicMessages
}

func (b openAIResponsesToAnthropicBridge) EncodeUpstreamRequest(req *LLMRequest, opts EncodeRequestOptions) ([]byte, error) {
	if req == nil {
		return nil, fmt.Errorf("encode openai responses to anthropic request: nil request")
	}

	model := req.Model
	if opts.Model != "" {
		model = opts.Model
	}

	request := anthropicRequest{
		Model:         model,
		Messages:      make([]anthropicMessage, 0, len(req.Prompt)),
		MaxTokens:     maxOutputTokensOrDefault(req.MaxOutputTokens),
		Temperature:   req.Temperature,
		StopSequences: req.StopSequences,
		TopP:          req.TopP,
		Tools:         encodeAnthropicTools(req.Tools),
		Stream:        req.Stream,
	}
	request.OutputConfig = encodeAnthropicOutputConfig(req.ResponseFormat)
	request.Thinking = encodeAnthropicThinkingForOpenAIInbound(req, request.MaxTokens)
	request.ToolChoice = encodeAnthropicToolChoice(sanitizeAnthropicToolChoice(req.ToolChoice, request.Thinking), req.ParallelToolCalls)
	for _, warning := range unsupportedAnthropicToolWarnings(req.Tools) {
		request.System = appendSystemText(request.System, "Proxy compatibility warning: "+warning)
	}

	previousWasTool := false
	for _, message := range req.Prompt {
		if message.Role == RoleSystem || message.Role == RoleDeveloper {
			request.System = appendSystemText(request.System, joinTextParts(message.Parts))
			previousWasTool = false
			continue
		}
		request.Messages, previousWasTool = appendOpenAIInboundAnthropicMessage(request.Messages, message, previousWasTool)
	}

	applyAnthropicCache(&request, req.Cache)
	return json.Marshal(request)
}

func (b openAIResponsesToAnthropicBridge) DecodeUpstreamResponse(raw []byte) (*LLMResponse, error) {
	resp, err := b.adapter.DecodeResponse(raw)
	if err != nil {
		return nil, err
	}
	resp.Protocol = ProtocolOpenAIResponses
	resp.Usage = anthropicUsageToResponsesUsage(resp.Usage)
	return resp, nil
}

func (b openAIResponsesToAnthropicBridge) NewStreamDecoder(opts StreamDecodeOptions) (StreamDecoder, error) {
	return (&AnthropicMessagesAdapter{}).NewStreamDecoder(opts)
}

func (b openAIResponsesToAnthropicBridge) NewStreamEncoder(opts StreamEncodeOptions) (StreamEncoder, error) {
	return &anthropicToOpenAIResponsesStreamEncoder{base: openAIResponsesStreamEncoder{model: opts.Model}}, nil
}

type anthropicToOpenAIResponsesStreamEncoder struct {
	base openAIResponsesStreamEncoder
}

func (e *anthropicToOpenAIResponsesStreamEncoder) Encode(part StreamPart) ([]RawStreamEvent, error) {
	if part.Type == StreamStart || part.Type == StreamFinish {
		part.Usage = anthropicUsageToResponsesUsage(part.Usage)
	}
	if part.Type == StreamReasoningStart {
		events, err := e.base.Encode(StreamPart{Type: StreamReasoningStart, ID: part.ID})
		if err != nil {
			return nil, err
		}
		if part.Delta != "" {
			deltaEvents, err := e.base.Encode(StreamPart{Type: StreamReasoningDelta, ID: part.ID, Delta: part.Delta})
			if err != nil {
				return nil, err
			}
			events = append(events, deltaEvents...)
		}
		return events, nil
	}
	if part.Type == StreamReasoningDelta {
		if signature, ok := part.ProviderMetadata["signature"].(string); ok && signature != "" && part.Delta == "" {
			return nil, nil
		}
	}
	return e.base.Encode(part)
}

func (e *anthropicToOpenAIResponsesStreamEncoder) Close() ([]RawStreamEvent, error) {
	return e.base.Close()
}

func (e *anthropicToOpenAIResponsesStreamEncoder) EncodeError(err error) []RawStreamEvent {
	return e.base.EncodeError(err)
}
