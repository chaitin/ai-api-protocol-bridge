package protocolbridge

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// A conversion loss must be reported on the request or response, never written
// into the request the model is asked about.
func TestChatEncoderReportsUrlOnlyDocument(t *testing.T) {
	adapter := NewOpenAIChatAdapter()
	req := &LLMRequest{
		Protocol: ProtocolOpenAIResponses,
		Model:    "gpt-4o",
		Prompt: []Message{{Role: RoleUser, Parts: []Part{
			{Type: PartText, Text: &TextPart{Text: "Summarise this"}},
			{Type: PartFile, File: &FilePart{Type: FileDocument, URL: "https://example.com/report.pdf", Filename: "report.pdf", MediaType: "application/pdf"}},
		}}},
	}

	raw, err := adapter.EncodeRequest(req, EncodeRequestOptions{Model: "gpt-4o"})
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}

	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	messages := decoded["messages"].([]any)
	content := messages[0].(map[string]any)["content"]
	// The text survives and the url-only document does not become an invented
	// block the model would read as part of the request.
	if content != "Summarise this" {
		t.Fatalf("content = %+v", content)
	}

	if len(req.Warnings) != 1 {
		t.Fatalf("warnings = %+v, want the dropped document reported", req.Warnings)
	}
	warning := req.Warnings[0]
	if warning.Code != LossUnsupportedFileInput {
		t.Fatalf("warning = %+v", warning)
	}
	if !strings.Contains(warning.Message, "report.pdf") {
		t.Fatalf("warning message = %q", warning.Message)
	}
	if warning.Path != "messages[0].content[1]" {
		t.Fatalf("warning path = %q", warning.Path)
	}
	if warning.From != ProtocolOpenAIResponses || warning.To != ProtocolOpenAIChat {
		t.Fatalf("warning directions = %s -> %s", warning.From, warning.To)
	}
	if warning.Severity != SeverityWarning {
		t.Fatalf("severity = %q", warning.Severity)
	}
}

func TestConversionErrorNamesEveryLoss(t *testing.T) {
	adapter := NewOpenAIChatAdapter()
	req := &LLMRequest{
		Protocol: ProtocolOpenAIResponses,
		Model:    "gpt-4o",
		Prompt: []Message{{Role: RoleUser, Parts: []Part{
			{Type: PartFile, File: &FilePart{Type: FileDocument, URL: "https://example.com/a.pdf", Filename: "a.pdf"}},
			{Type: PartFile, File: &FilePart{Type: FileImage, FileID: "file_9"}},
		}}},
	}

	_, err := adapter.EncodeRequest(req, EncodeRequestOptions{Model: "gpt-4o", LossPolicy: LossPolicyStrict})
	conversion, ok := err.(*ConversionError)
	if !ok {
		t.Fatalf("EncodeRequest() error = %v, want a *ConversionError", err)
	}
	if len(conversion.Warnings) != 2 {
		t.Fatalf("warnings = %+v", conversion.Warnings)
	}
	text := conversion.Error()
	for _, want := range []string{LossUnsupportedFileInput, "a.pdf", LossUnsupportedFileReference, "file_9", "messages[0].content[0]"} {
		if !strings.Contains(text, want) {
			t.Fatalf("error %q does not mention %q", text, want)
		}
	}
}

func TestLossPolicyLevels(t *testing.T) {
	cases := []struct {
		name    string
		policy  LossPolicy
		refuses bool
	}{
		{name: "allow records only", policy: LossPolicyAllow, refuses: false},
		{name: "safe refuses an error-severity loss", policy: LossPolicySafe, refuses: true},
		{name: "strict refuses any loss", policy: LossPolicyStrict, refuses: true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// A file loss is warning severity; a dropped tool is error severity.
			fileRequest := &LLMRequest{
				Model:  "claude",
				Prompt: []Message{{Role: RoleUser, Parts: []Part{{Type: PartFile, File: &FilePart{Type: FileDocument, FileID: "file_1"}}}}},
			}
			_, fileErr := NewAnthropicMessagesAdapter().EncodeRequest(fileRequest, EncodeRequestOptions{Model: "claude", LossPolicy: tc.policy})
			if tc.policy == LossPolicyStrict && fileErr == nil {
				t.Fatal("strict should refuse a warning-severity loss")
			}
			if tc.policy != LossPolicyStrict && fileErr != nil {
				t.Fatalf("policy %q refused a warning-severity loss: %v", tc.policy, fileErr)
			}

			toolRequest := &LLMRequest{
				Model:  "claude",
				Prompt: []Message{{Role: RoleUser, Parts: []Part{{Type: PartText, Text: &TextPart{Text: "hi"}}}}},
				Tools:  []Tool{{Type: ToolProviderDefined, Name: "web_search_preview"}},
			}
			_, toolErr := NewAnthropicMessagesAdapter().EncodeRequest(toolRequest, EncodeRequestOptions{Model: "claude", LossPolicy: tc.policy})
			if tc.refuses && toolErr == nil {
				t.Fatalf("policy %q should refuse a dropped tool", tc.policy)
			}
			if !tc.refuses && toolErr != nil {
				t.Fatalf("policy %q refused a dropped tool: %v", tc.policy, toolErr)
			}
		})
	}
}

// include is how the Responses protocol asks for extra output, and only the
// Responses protocol has the field. On the way to chat and to Anthropic the
// request is served without it, which is a degradation worth a note rather than a
// failure — a coding agent sends reasoning.encrypted_content on every request.
func TestUnsupportedIncludeIsReportedAndDropped(t *testing.T) {
	targets := []struct {
		name     string
		upstream Protocol
		field    string
	}{
		{"chat upstream", ProtocolOpenAIChat, "include"},
		{"anthropic upstream", ProtocolAnthropicMessages, "include"},
	}
	for _, tc := range targets {
		t.Run(tc.name, func(t *testing.T) {
			bridge, ok := NewCrossFamilyBridgeForProtocol(ProtocolOpenAIResponses, tc.upstream)
			if !ok {
				t.Fatalf("no bridge to %s", tc.upstream)
			}
			req, err := NewOpenAIResponsesAdapter().DecodeRequest([]byte(codexRequest))
			if err != nil {
				t.Fatalf("DecodeRequest() error = %v", err)
			}
			if len(req.Include) != 1 {
				t.Fatalf("fixture include = %+v", req.Include)
			}

			// Strict refuses anything that lost content. This is informational,
			// so the request must still go through.
			raw, err := bridge.EncodeUpstreamRequest(req, EncodeRequestOptions{Model: "upstream-model", LossPolicy: LossPolicyStrict})
			if err != nil {
				t.Fatalf("EncodeUpstreamRequest() under strict policy error = %v", err)
			}

			var decoded map[string]any
			if err := json.Unmarshal(raw, &decoded); err != nil {
				t.Fatalf("Unmarshal() error = %v", err)
			}
			if _, present := decoded[tc.field]; present {
				t.Fatalf("%s reached the upstream: %s", tc.field, raw)
			}

			found := false
			for _, warning := range req.Warnings {
				if warning.Code != LossUnsupportedInclude {
					continue
				}
				found = true
				if warning.Severity != SeverityInfo {
					t.Fatalf("severity = %q, want info", warning.Severity)
				}
				if warning.Path != "include[0]" {
					t.Fatalf("path = %q", warning.Path)
				}
				if warning.From != ProtocolOpenAIResponses || warning.To != tc.upstream {
					t.Fatalf("directions = %s -> %s", warning.From, warning.To)
				}
				if !strings.Contains(warning.Message, "reasoning.encrypted_content") {
					t.Fatalf("message = %q", warning.Message)
				}
			}
			if !found {
				t.Fatalf("warnings = %+v, want an include report", req.Warnings)
			}
		})
	}
}

// A chat request has no include field to lose, and blank entries are not a
// request for anything.
func TestUnsupportedIncludeIsNotReportedWithoutEntries(t *testing.T) {
	bridge, ok := NewCrossFamilyBridgeForProtocol(ProtocolOpenAIChat, ProtocolAnthropicMessages)
	if !ok {
		t.Fatal("no chat to anthropic bridge")
	}
	req, err := NewOpenAIChatAdapter().DecodeRequest([]byte(`{"model":"gpt-5.4","messages":[{"role":"user","content":"Hello"}]}`))
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	if _, err := bridge.EncodeUpstreamRequest(req, EncodeRequestOptions{Model: "claude"}); err != nil {
		t.Fatalf("EncodeUpstreamRequest() error = %v", err)
	}
	if len(req.Warnings) != 0 {
		t.Fatalf("warnings = %+v", req.Warnings)
	}

	// Whitespace is not an entry.
	req.Include = []string{"  ", ""}
	if _, err := bridge.EncodeUpstreamRequest(req, EncodeRequestOptions{Model: "claude"}); err != nil {
		t.Fatalf("EncodeUpstreamRequest() error = %v", err)
	}
	if len(req.Warnings) != 0 {
		t.Fatalf("warnings = %+v, want blank entries ignored", req.Warnings)
	}
}

// Anthropic Messages and OpenAI Responses each describe one assistant turn. A
// chat upstream asked for several completions has all but the first dropped, and
// the caller paid for them, so the loss is reported rather than absorbed.
func TestDroppedChoicesAreReported(t *testing.T) {
	chatResponse := []byte(`{
	  "id": "chatcmpl-1",
	  "object": "chat.completion",
	  "created": 1,
	  "model": "gpt-4o",
	  "choices": [
	    {"index": 0, "message": {"role": "assistant", "content": "first"}, "finish_reason": "stop"},
	    {"index": 1, "message": {"role": "assistant", "content": "second"}, "finish_reason": "stop"}
	  ],
	  "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}
	}`)

	cases := []struct {
		name    string
		adapter Adapter
	}{
		{"anthropic messages", NewAnthropicMessagesAdapter()},
		{"openai responses", NewOpenAIResponsesAdapter()},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			resp, err := NewOpenAIChatAdapter().DecodeResponse(chatResponse)
			if err != nil {
				t.Fatalf("DecodeResponse() error = %v", err)
			}
			resp.Protocol = ProtocolOpenAIChat
			if _, err := tc.adapter.EncodeResponse(resp, EncodeResponseOptions{}); err != nil {
				t.Fatalf("EncodeResponse() error = %v", err)
			}
			if len(resp.Warnings) != 1 {
				t.Fatalf("warnings = %+v", resp.Warnings)
			}
			warning := resp.Warnings[0]
			if warning.Code != LossDroppedChoice || warning.Severity != SeverityWarning {
				t.Fatalf("warning = %+v", warning)
			}
			if warning.Path != "choices" || !strings.Contains(warning.Message, "1 of the 2") {
				t.Fatalf("warning = %+v", warning)
			}
			if warning.From != ProtocolOpenAIChat {
				t.Fatalf("warning from = %q", warning.From)
			}
		})
	}

	// A single choice is not a loss.
	resp, err := NewOpenAIChatAdapter().DecodeResponse([]byte(`{"id":"c","choices":[{"index":0,"message":{"role":"assistant","content":"only"},"finish_reason":"stop"}]}`))
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	if _, err := NewAnthropicMessagesAdapter().EncodeResponse(resp, EncodeResponseOptions{}); err != nil {
		t.Fatalf("EncodeResponse() error = %v", err)
	}
	if len(resp.Warnings) != 0 {
		t.Fatalf("warnings = %+v, want none for a single choice", resp.Warnings)
	}
}

// A lost file must never be replaced by text the model reads as conversation.
// This guard fails if a converter reintroduces that, because the text changes
// the request bytes and invalidates the cached prompt prefix on every turn.
func TestNoConverterWritesWarningsIntoTheRequest(t *testing.T) {
	markers := []string{"Proxy warning", "Proxy compatibility warning"}

	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatalf("ReadDir() error = %v", err)
	}
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		source, err := os.ReadFile(filepath.Join(".", name))
		if err != nil {
			t.Fatalf("ReadFile(%s) error = %v", name, err)
		}
		for _, marker := range markers {
			if strings.Contains(string(source), marker) {
				t.Fatalf("%s writes %q into a request; report a Warning instead", name, marker)
			}
		}
	}
}
