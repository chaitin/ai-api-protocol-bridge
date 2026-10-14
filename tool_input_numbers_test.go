package protocolbridge

import (
	"encoding/json"
	"strings"
	"testing"
)

// toolInputNumbers are literals a float64 round trip changes: an integer past
// 2^53, an exponent, and a zero fraction. A tool argument such as a
// nanosecond timestamp or a snowflake id must reach the client as written.
var toolInputNumbers = []string{"123456789012345678", "2.5e3", "-7.0", "1.5"}

const toolInputJSON = `{"big":123456789012345678,"exp":2.5e3,"neg":-7.0,"ratio":1.5}`

func assertToolInputNumbers(t *testing.T, label, encoded string) {
	t.Helper()
	for _, literal := range toolInputNumbers {
		if !strings.Contains(encoded, literal) {
			t.Errorf("%s lost %s: %s", label, literal, encoded)
		}
	}
}

func TestToolCallInputKeepsItsNumbersAcrossProtocols(t *testing.T) {
	argumentsString, err := json.Marshal(toolInputJSON)
	if err != nil {
		t.Fatal(err)
	}
	responsesBody := `{"id":"resp_1","object":"response","status":"completed","model":"m","output":[{"type":"function_call","id":"fc_1","call_id":"call_1","name":"f","arguments":` + string(argumentsString) + `}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`
	chatBody := `{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"f","arguments":` + string(argumentsString) + `}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`
	anthropicBody := `{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[{"type":"tool_use","id":"toolu_1","name":"f","input":` + toolInputJSON + `}],"stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":1}}`

	tests := []struct {
		name     string
		inbound  Protocol
		upstream Protocol
		body     string
	}{
		{"responses upstream to chat client", ProtocolOpenAIChat, ProtocolOpenAIResponses, responsesBody},
		{"chat upstream to responses client", ProtocolOpenAIResponses, ProtocolOpenAIChat, chatBody},
		{"responses upstream to anthropic client", ProtocolAnthropicMessages, ProtocolOpenAIResponses, responsesBody},
		{"chat upstream to anthropic client", ProtocolAnthropicMessages, ProtocolOpenAIChat, chatBody},
		{"anthropic upstream to chat client", ProtocolOpenAIChat, ProtocolAnthropicMessages, anthropicBody},
		{"anthropic upstream to responses client", ProtocolOpenAIResponses, ProtocolAnthropicMessages, anthropicBody},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			bridge, ok := NewCrossFamilyBridgeForProtocol(tc.inbound, tc.upstream)
			if !ok {
				t.Fatalf("no bridge from %s to %s", tc.inbound, tc.upstream)
			}
			resp, err := bridge.DecodeUpstreamResponse([]byte(tc.body))
			if err != nil {
				t.Fatalf("decode: %v", err)
			}
			client, err := adapterFor(tc.inbound)
			if err != nil {
				t.Fatal(err)
			}
			encoded, err := client.EncodeResponse(resp, EncodeResponseOptions{Model: "m"})
			if err != nil {
				t.Fatalf("encode: %v", err)
			}
			assertToolInputNumbers(t, tc.name, string(encoded))
		})
	}
}

func adapterFor(protocol Protocol) (Adapter, error) {
	switch protocol {
	case ProtocolOpenAIChat:
		return NewOpenAIChatAdapter(), nil
	case ProtocolOpenAIResponses:
		return NewOpenAIResponsesAdapter(), nil
	default:
		return NewAnthropicMessagesAdapter(), nil
	}
}

// The same decoders read the tool calls in a request's history, so a prior
// turn's arguments reach the upstream as written too.
func TestToolCallInputKeepsItsNumbersInRequestHistory(t *testing.T) {
	request := `{"model":"m","max_tokens":16,"messages":[` +
		`{"role":"user","content":"go"},` +
		`{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"f","input":` + toolInputJSON + `}]},` +
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"done"}]}]}`
	decoded, err := NewAnthropicMessagesAdapter().DecodeRequest([]byte(request))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	for _, upstream := range []Protocol{ProtocolOpenAIChat, ProtocolOpenAIResponses} {
		bridge, ok := NewCrossFamilyBridgeForProtocol(ProtocolAnthropicMessages, upstream)
		if !ok {
			t.Fatalf("no bridge to %s", upstream)
		}
		encoded, err := bridge.EncodeUpstreamRequest(decoded, EncodeRequestOptions{Model: "m"})
		if err != nil {
			t.Fatalf("encode for %s: %v", upstream, err)
		}
		assertToolInputNumbers(t, string(upstream)+" request", string(encoded))
	}
}

func TestDecodeOpenAIToolInputKeepsTextThatIsNotOneJSONValue(t *testing.T) {
	for _, arguments := range []string{`not json`, `{"a":1} trailing`, `{"a":1}{"b":2}`} {
		input, err := decodeOpenAIToolInput(arguments)
		if err != nil || input != arguments {
			t.Errorf("decodeOpenAIToolInput(%q) = %#v, %v; want the text unchanged", arguments, input, err)
		}
	}
	input, err := decodeOpenAIToolInput(`{"big":123456789012345678}`)
	if err != nil {
		t.Fatal(err)
	}
	object, ok := input.(map[string]any)
	if !ok || object["big"] != json.Number("123456789012345678") {
		t.Errorf("decoded input = %#v, want the number kept as json.Number", input)
	}
}
