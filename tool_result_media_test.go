package protocolbridge

import (
	"encoding/json"
	"testing"
)

// A chat tool message carries text only. When an Anthropic client such as
// Claude Code returns an image from a tool — a screenshot, or a file it read
// from disk — the content arrives as an image block inside tool_result. The
// encoder used to flatten that content to its text, so the model received an
// empty tool result and the image silently vanished.
//
// These tests drive the real bridge with traffic-shaped input and assert the
// media survives into a following user message.

const anthropicToolResultImageRequest = `{
  "model": "claude-sonnet-4",
  "max_tokens": 8192,
  "tools": [{"name":"Read","description":"Read a file","input_schema":{"type":"object","properties":{"file_path":{"type":"string"}},"required":["file_path"]}}],
  "messages": [
    {"role":"user","content":"Read the screenshot"},
    {"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"Read","input":{"file_path":"/tmp/shot.png"}}]},
    {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":[
      {"type":"text","text":"Read image file"},
      {"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBORw0KGgo="}}
    ]}]}
  ]
}`

func anthropicToChatMessages(t *testing.T, request string) []map[string]any {
	t.Helper()

	decoded, err := NewAnthropicMessagesAdapter().DecodeRequest([]byte(request))
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	bridge, ok := NewCrossFamilyBridgeForProtocol(ProtocolAnthropicMessages, ProtocolOpenAIChat)
	if !ok {
		t.Fatal("no anthropic-to-chat bridge")
	}
	encoded, err := bridge.EncodeUpstreamRequest(decoded, EncodeRequestOptions{Model: "gpt-4o"})
	if err != nil {
		t.Fatalf("EncodeUpstreamRequest() error = %v", err)
	}

	var wire struct {
		Messages []map[string]any `json:"messages"`
	}
	if err := json.Unmarshal(encoded, &wire); err != nil {
		t.Fatalf("Unmarshal(%s) error = %v", encoded, err)
	}
	return wire.Messages
}

func TestAnthropicToolResultImageIsLiftedIntoFollowingUserMessage(t *testing.T) {
	messages := anthropicToChatMessages(t, anthropicToolResultImageRequest)

	roles := make([]string, 0, len(messages))
	for _, message := range messages {
		roles = append(roles, message["role"].(string))
	}
	want := []string{"user", "assistant", "tool", "user"}
	if len(roles) != len(want) {
		t.Fatalf("roles = %v, want %v", roles, want)
	}
	for i := range want {
		if roles[i] != want[i] {
			t.Fatalf("roles = %v, want %v", roles, want)
		}
	}

	toolMessage := messages[2]
	if toolMessage["tool_call_id"] != "toolu_1" {
		t.Fatalf("tool_call_id = %v", toolMessage["tool_call_id"])
	}
	if toolMessage["content"] != "Read image file" {
		t.Fatalf("tool content = %v, want the text part", toolMessage["content"])
	}

	content, ok := messages[3]["content"].([]any)
	if !ok {
		t.Fatalf("lifted user content = %#v, want a content array", messages[3]["content"])
	}
	if len(content) != 1 {
		t.Fatalf("lifted content = %+v", content)
	}
	part, ok := content[0].(map[string]any)
	if !ok || part["type"] != "image_url" {
		t.Fatalf("lifted part = %+v", content[0])
	}
	imageURL, ok := part["image_url"].(map[string]any)
	if !ok {
		t.Fatalf("image_url = %#v", part["image_url"])
	}
	if imageURL["url"] != "data:image/png;base64,iVBORw0KGgo=" {
		t.Fatalf("image url = %v", imageURL["url"])
	}
}

func TestAnthropicToolResultTextOnlyStillProducesOneToolMessage(t *testing.T) {
	messages := anthropicToChatMessages(t, `{
	  "model": "claude-sonnet-4",
	  "max_tokens": 8192,
	  "messages": [
	    {"role":"assistant","content":[{"type":"tool_use","id":"toolu_9","name":"Bash","input":{"command":"ls"}}]},
	    {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_9","content":"main.go"}]}
	  ]
	}`)

	roles := make([]string, 0, len(messages))
	for _, message := range messages {
		roles = append(roles, message["role"].(string))
	}
	want := []string{"assistant", "tool"}
	if len(roles) != len(want) || roles[0] != want[0] || roles[1] != want[1] {
		t.Fatalf("roles = %v, want %v: no extra user turn for a text-only result", roles, want)
	}
	if messages[1]["content"] != "main.go" {
		t.Fatalf("tool content = %v", messages[1]["content"])
	}
}

func TestChatToolMessageWithImageSurvivesDecoding(t *testing.T) {
	decoded, err := NewOpenAIChatAdapter().DecodeRequest([]byte(`{
	  "model": "gpt-4o",
	  "messages": [
	    {"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"Read","arguments":"{}"}}]},
	    {"role":"tool","tool_call_id":"call_1","content":[{"type":"text","text":"Read image file"},{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgo="}}]}
	  ]
	}`))
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}

	toolResult := findToolResultPart(decoded.Prompt)
	if toolResult == nil {
		t.Fatal("no tool result part")
	}
	if toolResult.Output.Type != ToolResultContent {
		t.Fatalf("output type = %q, want %q so the image is not flattened away", toolResult.Output.Type, ToolResultContent)
	}
	if len(toolResult.Output.Content) != 2 {
		t.Fatalf("output content = %+v", toolResult.Output.Content)
	}
}

func findToolResultPart(messages []Message) *ToolResultPart {
	for _, message := range messages {
		for _, part := range message.Parts {
			if part.Type == PartToolResult && part.ToolResult != nil {
				return part.ToolResult
			}
		}
	}
	return nil
}
