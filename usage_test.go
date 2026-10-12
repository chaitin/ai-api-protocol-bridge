package protocolbridge

import (
	"encoding/json"
	"testing"
)

// Anthropic reports prompt tokens on message_start and then reports only the
// output count on message_delta. A gateway that repeats input_tokens as 0 in
// that second event must not erase what the stream already reported: the host
// would bill nothing for the prompt.
func TestAnthropicStreamKeepsInputTokensReportedAsZeroLater(t *testing.T) {
	adapter := NewAnthropicMessagesAdapter()
	decoder, err := adapter.NewStreamDecoder(StreamDecodeOptions{})
	if err != nil {
		t.Fatalf("NewStreamDecoder() error = %v", err)
	}
	encoder, err := adapter.NewStreamEncoder(StreamEncodeOptions{Model: "claude-sonnet-4"})
	if err != nil {
		t.Fatalf("NewStreamEncoder() error = %v", err)
	}

	events := []string{
		`{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4","content":[],"usage":{"input_tokens":12000,"output_tokens":0,"cache_read_input_tokens":8000}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}`,
		`{"type":"content_block_stop","index":0}`,
		// The gateway restates input_tokens as 0 rather than omitting it.
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":0,"output_tokens":42}}`,
		`{"type":"message_stop"}`,
	}

	var out []RawStreamEvent
	for _, body := range events {
		var envelope struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal([]byte(body), &envelope); err != nil {
			t.Fatalf("Unmarshal(%s) error = %v", body, err)
		}
		parts, err := decoder.Decode(RawStreamEvent{Event: envelope.Type, Data: []byte(body)})
		if err != nil {
			t.Fatalf("Decode(%s) error = %v", body, err)
		}
		for _, part := range parts {
			encoded, err := encoder.Encode(part)
			if err != nil {
				t.Fatalf("Encode(%s) error = %v", part.Type, err)
			}
			out = append(out, encoded...)
		}
	}

	var usage *anthropicUsage
	for _, event := range out {
		var decoded anthropicStreamEvent
		if err := json.Unmarshal(event.Data, &decoded); err != nil {
			t.Fatalf("Unmarshal(%s) error = %v", event.Data, err)
		}
		if decoded.Type == "message_delta" && decoded.Usage != nil {
			usage = decoded.Usage
		}
	}
	if usage == nil {
		t.Fatalf("no message_delta carrying usage: %+v", out)
	}
	if usage.InputTokens == nil || *usage.InputTokens != 12000 {
		t.Fatalf("input_tokens = %v, want the 12000 the stream reported first", usage.InputTokens)
	}
	if usage.OutputTokens == nil || *usage.OutputTokens != 42 {
		t.Fatalf("output_tokens = %v", usage.OutputTokens)
	}
	if usage.CacheReadInputTokens == nil || *usage.CacheReadInputTokens != 8000 {
		t.Fatalf("cache_read_input_tokens = %v", usage.CacheReadInputTokens)
	}
}

func TestMergeUsagePrefersKnownCountsOverRestatedZeros(t *testing.T) {
	nonZero, zero := 12000, 0
	eight, seven := 8, 7

	cases := []struct {
		name   string
		base   Usage
		update Usage
		want   Usage
	}{
		{
			name:   "a restated zero does not erase a known count",
			base:   Usage{InputTokens: &nonZero},
			update: Usage{InputTokens: &zero},
			want:   Usage{InputTokens: &nonZero},
		},
		{
			name:   "an omitted field keeps its value",
			base:   Usage{InputTokens: &nonZero, OutputTokens: &eight},
			update: Usage{OutputTokens: &seven},
			want:   Usage{InputTokens: &nonZero, OutputTokens: &seven},
		},
		{
			name:   "a real count overwrites a known one",
			base:   Usage{OutputTokens: &eight},
			update: Usage{OutputTokens: &seven},
			want:   Usage{OutputTokens: &seven},
		},
		{
			name:   "a zero is recorded when nothing is known",
			base:   Usage{},
			update: Usage{InputTokens: &zero},
			want:   Usage{InputTokens: &zero},
		},
		{
			name:   "a zero replaces an earlier zero",
			base:   Usage{InputTokens: &zero},
			update: Usage{InputTokens: &zero},
			want:   Usage{InputTokens: &zero},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			base := tc.base
			mergeUsage(&base, tc.update)
			if !sameUsage(base, tc.want) {
				t.Fatalf("mergeUsage() = %+v, want %+v", base, tc.want)
			}
		})
	}
}

func sameUsage(left, right Usage) bool {
	fields := []struct{ l, r *int }{
		{left.InputTokens, right.InputTokens},
		{left.OutputTokens, right.OutputTokens},
		{left.ReasoningTokens, right.ReasoningTokens},
		{left.CachedInputTokens, right.CachedInputTokens},
		{left.CacheCreationInputTokens, right.CacheCreationInputTokens},
		{left.CacheReadInputTokens, right.CacheReadInputTokens},
	}
	for _, field := range fields {
		if (field.l == nil) != (field.r == nil) {
			return false
		}
		if field.l != nil && *field.l != *field.r {
			return false
		}
	}
	return true
}

// The two usage conversions are inverses, which is what keeps a bridged bill
// equal to the bill the upstream would have produced. Reading the raw fields
// instead of converting is how a cached prompt gets counted twice.
func TestUsageConversionsRoundTrip(t *testing.T) {
	t.Run("anthropic to openai and back", func(t *testing.T) {
		// Anthropic terms: input excludes the cache, which is reported apart.
		input, cacheRead, output := 12000, 8000, 42
		anthropic := Usage{InputTokens: &input, CacheReadInputTokens: &cacheRead, CachedInputTokens: &cacheRead, OutputTokens: &output}

		openai := anthropicUsageToResponsesUsage(anthropic)
		if openai.InputTokens == nil || *openai.InputTokens != 20000 {
			t.Fatalf("openai input_tokens = %v, want the 20000 total", openai.InputTokens)
		}
		if openai.CachedInputTokens == nil || *openai.CachedInputTokens != 8000 {
			t.Fatalf("openai cached = %v", openai.CachedInputTokens)
		}
		// OpenAI terms bill the total with the cache broken out, so this is the
		// same money as the Anthropic view.
		openaiBilling := billingUsageForProtocol(ProtocolOpenAIResponses, openai)
		if openaiBilling.InputTokens != 12000 || openaiBilling.CachedInputTokens != 8000 {
			t.Fatalf("openai billing = %+v", openaiBilling)
		}

		back := responsesUsageToAnthropicUsage(openai)
		anthropicBilling := billingUsageForProtocol(ProtocolAnthropicMessages, back)
		if anthropicBilling.InputTokens != openaiBilling.InputTokens || anthropicBilling.CachedInputTokens != openaiBilling.CachedInputTokens {
			t.Fatalf("round trip billing = %+v, want %+v", anthropicBilling, openaiBilling)
		}
	})

	t.Run("openai to anthropic and back", func(t *testing.T) {
		input, cached, output := 20000, 8000, 42
		openai := Usage{InputTokens: &input, CachedInputTokens: &cached, OutputTokens: &output}

		anthropic := responsesUsageToAnthropicUsage(openai)
		if anthropic.InputTokens == nil || *anthropic.InputTokens != 12000 {
			t.Fatalf("anthropic input_tokens = %v, want the 12000 that excludes the cache", anthropic.InputTokens)
		}

		back := anthropicUsageToResponsesUsage(anthropic)
		if back.InputTokens == nil || *back.InputTokens != 20000 {
			t.Fatalf("round trip input_tokens = %v", back.InputTokens)
		}
		openaiBilling := billingUsageForProtocol(ProtocolOpenAIResponses, back)
		if openaiBilling.InputTokens != 12000 || openaiBilling.CachedInputTokens != 8000 {
			t.Fatalf("round trip billing = %+v", openaiBilling)
		}
	})
}
