package protocolbridge

import "fmt"

// anthropicToOpenAIChatBridge serves an Anthropic-messages client against an
// OpenAI chat-completions upstream.
//
// It is the missing fourth cross-family cell. An Anthropic inbound has two
// possible OpenAI targets, so a caller that knows which one its upstream speaks
// must be able to ask for it by protocol rather than by family — see
// NewCrossFamilyBridgeForProtocol.
//
// The upstream half needs no hand-written encoder here: the caller has already
// decoded the Anthropic request into the neutral LLMRequest, and the chat
// adapter's own encoder is the authority on that wire format. Hand-building it,
// as the Responses bridge must, would only duplicate the adapter.
type anthropicToOpenAIChatBridge struct {
	adapter OpenAIChatAdapter
}

func (b anthropicToOpenAIChatBridge) InboundProtocol() Protocol {
	return ProtocolAnthropicMessages
}

func (b anthropicToOpenAIChatBridge) UpstreamProtocol() Protocol {
	return ProtocolOpenAIChat
}

func (b anthropicToOpenAIChatBridge) EncodeUpstreamRequest(req *LLMRequest, opts EncodeRequestOptions) ([]byte, error) {
	if req == nil {
		return nil, fmt.Errorf("encode anthropic to openai chat request: nil request")
	}
	return b.adapter.EncodeRequest(req, opts)
}

func (b anthropicToOpenAIChatBridge) DecodeUpstreamResponse(raw []byte) (*LLMResponse, error) {
	resp, err := b.adapter.DecodeResponse(raw)
	if err != nil {
		return nil, err
	}
	resp.Protocol = ProtocolOpenAIChat
	return resp, nil
}

func (b anthropicToOpenAIChatBridge) NewStreamDecoder(opts StreamDecodeOptions) (StreamDecoder, error) {
	return b.adapter.NewStreamDecoder(opts)
}

func (b anthropicToOpenAIChatBridge) NewStreamEncoder(opts StreamEncodeOptions) (StreamEncoder, error) {
	return &anthropicStreamEncoderForChatUpstream{ant: newAnthropicStreamEncoder(opts)}, nil
}

// anthropicStreamEncoderForChatUpstream writes neutral stream parts as Anthropic
// SSE for a chat-completions upstream.
//
// Two things separate it from the plain Anthropic encoder.
//
// The first is usage. The chat decoder reports prompt tokens the way OpenAI
// does — the total, with the cached portion broken out separately — while
// Anthropic expects input_tokens to exclude what was served from cache and
// reports the cached count in cache_read_input_tokens. Passing the chat numbers
// straight through would count every cached token twice.
//
// The second is ordering. OpenAI reports usage in a chunk of its own, after the
// chunk carrying finish_reason, and the chat decoder surfaces it as a
// StreamResponseMetadata part. Anthropic has no event that can carry usage after
// message_delta, and the plain encoder ignores that part outright, so a finish
// that arrives without usage is held until the usage arrives. A stream that
// never reports usage still gets its finish: Close flushes it.
type anthropicStreamEncoderForChatUpstream struct {
	ant     anthropicStreamEncoder
	pending *StreamPart
}

func (e *anthropicStreamEncoderForChatUpstream) Encode(part StreamPart) ([]RawStreamEvent, error) {
	switch part.Type {
	case StreamResponseMetadata:
		return e.flushPending(part.Usage)
	case StreamFinish:
		// A provider may report usage on the finish chunk itself, in which case
		// there is nothing to wait for.
		if hasUsage(part.Usage) {
			part.Usage = responsesUsageToAnthropicUsage(part.Usage)
			return e.ant.Encode(part)
		}
		held := part
		e.pending = &held
		return nil, nil
	case StreamStart:
		part.Usage = responsesUsageToAnthropicUsage(part.Usage)
	}
	if part.Type == StreamReasoningDelta {
		if signature, ok := part.ProviderMetadata["signature"].(string); ok && signature != "" && part.Delta == "" {
			return e.ant.Encode(StreamPart{Type: StreamReasoningDelta, ID: part.ID, ProviderMetadata: map[string]any{"signature": signature}})
		}
	}
	return e.ant.Encode(part)
}

// flushPending emits a held finish, carrying the usage that has just arrived.
func (e *anthropicStreamEncoderForChatUpstream) flushPending(usage Usage) ([]RawStreamEvent, error) {
	if e.pending == nil {
		return nil, nil
	}
	finish := *e.pending
	e.pending = nil
	if hasUsage(usage) {
		finish.Usage = responsesUsageToAnthropicUsage(usage)
	}
	return e.ant.Encode(finish)
}

func (e *anthropicStreamEncoderForChatUpstream) Close() ([]RawStreamEvent, error) {
	events, err := e.flushPending(Usage{})
	if err != nil {
		return nil, err
	}
	closed, err := e.ant.Close()
	if err != nil {
		return nil, err
	}
	return append(events, closed...), nil
}

func (e *anthropicStreamEncoderForChatUpstream) EncodeError(err error) []RawStreamEvent {
	return e.ant.EncodeError(err)
}
