package protocolbridge

import "fmt"

// openAIResponsesToOpenAIChatBridge serves an OpenAI Responses client — codex —
// against an OpenAI chat-completions upstream.
//
// It is the cell that matters most in practice: a Responses-capable client is
// the common case, chat completions is what most compatible providers speak,
// and without this pair the caller has no bridge at all.
//
// Like the Anthropic to chat bridge, the upstream half needs no hand-written
// encoder: the caller has already decoded the Responses request into the
// neutral LLMRequest, and the chat adapter is the authority on that wire
// format.
type openAIResponsesToOpenAIChatBridge struct {
	adapter OpenAIChatAdapter
}

func (b openAIResponsesToOpenAIChatBridge) InboundProtocol() Protocol {
	return ProtocolOpenAIResponses
}

func (b openAIResponsesToOpenAIChatBridge) UpstreamProtocol() Protocol {
	return ProtocolOpenAIChat
}

func (b openAIResponsesToOpenAIChatBridge) EncodeUpstreamRequest(req *LLMRequest, opts EncodeRequestOptions) ([]byte, error) {
	if req == nil {
		return nil, fmt.Errorf("encode openai responses to openai chat request: nil request")
	}
	return b.adapter.EncodeRequest(req, opts)
}

func (b openAIResponsesToOpenAIChatBridge) DecodeUpstreamResponse(raw []byte) (*LLMResponse, error) {
	resp, err := b.adapter.DecodeResponse(raw)
	if err != nil {
		return nil, err
	}
	resp.Protocol = ProtocolOpenAIChat
	return resp, nil
}

func (b openAIResponsesToOpenAIChatBridge) NewStreamDecoder(opts StreamDecodeOptions) (StreamDecoder, error) {
	return b.adapter.NewStreamDecoder(opts)
}

func (b openAIResponsesToOpenAIChatBridge) NewStreamEncoder(opts StreamEncodeOptions) (StreamEncoder, error) {
	return &responsesStreamEncoderForChatUpstream{res: openAIResponsesStreamEncoder{model: opts.Model}}, nil
}

// responsesStreamEncoderForChatUpstream writes neutral stream parts as OpenAI
// Responses SSE for a chat-completions upstream.
//
// The reason it exists is ordering, and it is the mirror of the Anthropic
// bridge's problem. In a chat stream, usage arrives in a chunk of its own after
// the chunk carrying finish_reason, and the chat decoder surfaces it as a
// StreamResponseMetadata part; the Responses encoder ignores that part. But
// response.completed is the only Responses event that can carry usage, and it
// is emitted from the finish. Encoding the finish the moment it arrives would
// publish the response as complete with no token counts at all, so a finish
// that arrives without usage is held until the usage shows up. A stream that
// never reports usage still gets its finish: Close flushes it.
//
// No usage conversion is needed here, unlike in the Anthropic direction: chat
// and Responses both count prompt tokens the OpenAI way, the total with the
// cached portion broken out separately.
type responsesStreamEncoderForChatUpstream struct {
	res     openAIResponsesStreamEncoder
	pending *StreamPart

	// lastUsage keeps a usage report that arrived before the finish, so a
	// finish that carries none can still be completed with it.
	lastUsage Usage
}

func (e *responsesStreamEncoderForChatUpstream) Encode(part StreamPart) ([]RawStreamEvent, error) {
	switch part.Type {
	case StreamResponseMetadata:
		if hasUsage(part.Usage) {
			e.lastUsage = part.Usage
		}
		return e.flushPending(part.Usage)
	case StreamFinish:
		if !hasUsage(part.Usage) && hasUsage(e.lastUsage) {
			part.Usage = e.lastUsage
		}
		if hasUsage(part.Usage) {
			return e.res.Encode(part)
		}
		held := part
		e.pending = &held
		return nil, nil
	}
	return e.res.Encode(part)
}

// flushPending emits a held finish, carrying the usage that has just arrived.
func (e *responsesStreamEncoderForChatUpstream) flushPending(usage Usage) ([]RawStreamEvent, error) {
	if e.pending == nil {
		return nil, nil
	}
	finish := *e.pending
	e.pending = nil
	if hasUsage(usage) {
		finish.Usage = usage
	} else if hasUsage(e.lastUsage) {
		finish.Usage = e.lastUsage
	}
	return e.res.Encode(finish)
}

func (e *responsesStreamEncoderForChatUpstream) Close() ([]RawStreamEvent, error) {
	events, err := e.flushPending(Usage{})
	if err != nil {
		return nil, err
	}
	closed, err := e.res.Close()
	if err != nil {
		return nil, err
	}
	return append(events, closed...), nil
}

func (e *responsesStreamEncoderForChatUpstream) EncodeError(err error) []RawStreamEvent {
	return e.res.EncodeError(err)
}
