package util

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/sashabaranov/go-openai"
)

func TestAnthropicPromptCacheBreakpointsAndSchema(t *testing.T) {
	schema := json.RawMessage(`{"type":"object","properties":{"action":{"type":"string"}},
		"allOf":[{"if":{"properties":{"action":{"const":"get"}}},"then":{"required":["id"]}}]}`)
	request := openai.ChatCompletionRequest{
		Model: "claude-opus-5-5", MaxCompletionTokens: 1024,
		Tools: []openai.Tool{{Type: openai.ToolTypeFunction, Function: &openai.FunctionDefinition{Name: "snippet", Parameters: schema}}},
		Messages: []openai.ChatCompletionMessage{
			{Role: openai.ChatMessageRoleSystem, Content: "system prompt"},
			{Role: openai.ChatMessageRoleUser, Content: "first"},
			{Role: openai.ChatMessageRoleAssistant, ToolCalls: []openai.ToolCall{{ID: "c1", Type: openai.ToolTypeFunction,
				Function: openai.FunctionCall{Name: "snippet", Arguments: `{"action":"list"}`}}}},
			{Role: openai.ChatMessageRoleTool, ToolCallID: "c1", Content: "result"},
			{Role: openai.ChatMessageRoleAssistant, Content: "answer"},
			{Role: openai.ChatMessageRoleUser, Content: "second"},
		},
	}
	ret, err := buildAnthropicRequest(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	body, _ := json.Marshal(ret)
	if got := strings.Count(string(body), `"cache_control"`); got != 3 {
		t.Fatalf("expected 3 cache breakpoints, got %d: %s", got, body)
	}
	if strings.Contains(string(body), `"allOf"`) {
		t.Fatalf("top-level allOf must be stripped from input_schema: %s", body)
	}
	if !strings.Contains(string(ret.System[0]), "cache_control") {
		t.Fatal("system block must carry a cache breakpoint")
	}
	lastUser := ret.Messages[len(ret.Messages)-1]
	if !strings.Contains(string(lastUser.Content[len(lastUser.Content)-1]), "cache_control") {
		t.Fatal("last message must carry a cache breakpoint")
	}
	// 上一条 user 消息（tool_result）也应带断点，原始 schema 不得被修改。
	if !strings.Contains(string(ret.Messages[len(ret.Messages)-3].Content[0]), "cache_control") {
		t.Fatalf("previous user message must carry a cache breakpoint: %s", body)
	}
	if !strings.Contains(string(schema), "allOf") {
		t.Fatal("original tool schema must not be mutated")
	}
}

func TestAnthropicCacheBreakpointSkipsThinking(t *testing.T) {
	if _, ok := anthropicWithCacheControl(json.RawMessage(`{"type":"thinking","thinking":"x","signature":"s"}`)); ok {
		t.Fatal("thinking block must not get cache_control")
	}
	if _, ok := anthropicWithCacheControl(json.RawMessage(`{"type":"text","text":" "}`)); ok {
		t.Fatal("blank text block must not get cache_control")
	}
}
