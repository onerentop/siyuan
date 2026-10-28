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

func TestAnthropicInputSchemaNormalization(t *testing.T) {
	marshal := func(value any) string {
		data, _ := json.Marshal(value)
		return string(data)
	}

	// 合法 schema 原样返回，保持请求前缀稳定。
	plain := json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}},"required":["q"]}`)
	if got := anthropicInputSchema(plain); marshal(got) != string(plain) {
		t.Fatalf("valid schema must pass through unchanged: %s", marshal(got))
	}

	// 参数只定义在顶层 anyOf 变体中：合并属性、补 type，并保留 $defs 引用。
	rootAnyOf := map[string]any{
		"$defs": map[string]any{"id": map[string]any{"type": "string"}},
		"anyOf": []any{
			map[string]any{"type": "object", "properties": map[string]any{"id": map[string]any{"$ref": "#/$defs/id"}}, "required": []any{"id"}},
			map[string]any{"type": "object", "properties": map[string]any{"path": map[string]any{"type": "string"}}},
		},
	}
	got, _ := anthropicInputSchema(rootAnyOf).(map[string]any)
	properties, _ := got["properties"].(map[string]any)
	if got["type"] != "object" || len(properties) != 2 || properties["id"] == nil || properties["path"] == nil {
		t.Fatalf("variant properties must be merged under type=object: %s", marshal(got))
	}
	if _, ok := got["anyOf"]; ok {
		t.Fatalf("top-level anyOf must be removed: %s", marshal(got))
	}
	if _, ok := got["$defs"]; !ok {
		t.Fatalf("$defs must be preserved for $ref: %s", marshal(got))
	}
	if _, ok := rootAnyOf["anyOf"]; !ok {
		t.Fatal("original schema must not be mutated")
	}

	// 缺 type 与 properties 的空 schema 也要满足 Messages API 的顶层要求。
	empty, _ := anthropicInputSchema(map[string]any{}).(map[string]any)
	if empty["type"] != "object" || empty["properties"] == nil {
		t.Fatalf("empty schema must become an object schema: %s", marshal(empty))
	}
}

func TestAnthropicCacheBreakpointPreservesNumbers(t *testing.T) {
	block, ok := anthropicWithCacheControl(json.RawMessage(`{"type":"text","text":"x","n":9007199254740993}`))
	if !ok || !strings.Contains(string(block), "9007199254740993") {
		t.Fatalf("large integers must survive the cache_control rewrite: %s", block)
	}
}
