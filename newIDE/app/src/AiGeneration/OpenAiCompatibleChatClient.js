// @flow

// Client for OpenAI-compatible Chat Completions endpoints — the foundation
// for letting users bring their own provider (for example IO Intelligence by
// io.net) instead of the hosted GDevelop generation service.
//
// Only translation and transport live here: the editor's AI request records
// (see the types in Utils/GDevelopServices/Generation.js) are translated to
// Chat Completions messages and back. Wiring this into the AI request
// lifecycle (preferences, routing, UI) is deliberately NOT part of this
// module so any custom-provider integration can reuse it.

import axios from 'axios';
import type {
  AiRequestMessage,
  AiRequestUserMessage,
  AiRequestAssistantMessage,
  AiRequestFunctionCallOutput,
} from '../Utils/GDevelopServices/Generation';

export type OpenAiCompatibleChatConfiguration = {|
  baseUrl: string,
  apiKey: string,
  model: string,
|};

// Pre-filled defaults for IO Intelligence (io.net), an OpenAI-compatible
// Chat Completions API. Model ids are HF-style `org/name`; the live list is
// `GET {baseUrl}/models` (e.g. meta-llama/Llama-3.3-70B-Instruct).
export const IO_INTELLIGENCE_DEFAULTS = {
  baseUrl: 'https://api.intelligence.io.solutions/api/v1',
  model: 'meta-llama/Llama-3.3-70B-Instruct',
};

export type ChatCompletionsTool = {
  type: 'function',
  function: {
    name: string,
    description?: string,
    parameters?: Object,
  },
};

export type ChatCompletionsToolCall = {
  id: string,
  type: 'function',
  function: { name: string, arguments: string },
};

export type ChatCompletionsMessage = {
  role: 'system' | 'user' | 'assistant' | 'tool',
  content: string | null,
  tool_calls?: Array<ChatCompletionsToolCall>,
  tool_call_id?: string,
};

// Content items of an assistant record, in the shape used by
// AiRequestAssistantMessage content entries (output_text / function_call).
export type ChatCompletionsResponseContentItem =
  | {
      type: 'output_text',
      status: 'completed',
      text: string,
      annotations: Array<{}>,
    }
  | {
      type: 'function_call',
      status: 'completed',
      call_id: string,
      name: string,
      arguments: string,
    };

export type ChatCompletionsTransportRequest = {|
  url: string,
  headers: { [name: string]: string },
  body: Object,
|};

export type ChatCompletionsTransportResponse = {|
  status: number,
  data: Object | string | null,
|};

// A transport performs the HTTP request. It resolves for EVERY HTTP status
// (error handling for non-2xx statuses lives in sendChatCompletion) and must
// serialize the body object as JSON — the default transport lets axios do it.
// Network-level failures (DNS, connection reset, timeout) reject.
export type ChatCompletionsTransport = (
  request: ChatCompletionsTransportRequest
) => Promise<ChatCompletionsTransportResponse>;

// Joins a base URL and a path, tolerating trailing slashes on the base and a
// missing leading slash on the path:
// joinUrl('https://x/api/v1', 'models') === 'https://x/api/v1/models'.
export const joinUrl = (baseUrl: string, path: string): string =>
  baseUrl.replace(/\/+$/, '') + (path.startsWith('/') ? path : `/${path}`);

// --- GDevelop AI request records -> Chat Completions messages ---

export const translateUserMessage = (
  message: AiRequestUserMessage
): ChatCompletionsMessage => ({
  role: 'user',
  content: message.content.map(content => content.text).join('\n\n'),
});

export const translateAssistantMessage = (
  message: AiRequestAssistantMessage
): ChatCompletionsMessage => {
  const textParts: Array<string> = [];
  const toolCalls: Array<ChatCompletionsToolCall> = [];
  message.content.forEach(content => {
    if (content.type === 'output_text') {
      textParts.push(content.text);
    } else if (content.type === 'function_call') {
      toolCalls.push({
        id: content.call_id,
        type: 'function',
        function: { name: content.name, arguments: content.arguments },
      });
    }
    // 'reasoning' content is not representable in Chat Completions and is
    // deliberately dropped.
  });
  return {
    role: 'assistant',
    content: textParts.length > 0 ? textParts.join('\n\n') : null,
    ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
  };
};

export const translateFunctionCallOutput = (
  message: AiRequestFunctionCallOutput
): ChatCompletionsMessage => ({
  role: 'tool',
  content: message.output,
  tool_call_id: message.call_id,
});

export const translateAiRequestMessagesToChatMessages = (
  messages: Array<AiRequestMessage>
): Array<ChatCompletionsMessage> =>
  messages.map(message => {
    if (message.type === 'message') {
      return message.role === 'user'
        ? translateUserMessage(message)
        : translateAssistantMessage(message);
    }
    return translateFunctionCallOutput(message);
  });

// --- Chat Completions response -> GDevelop assistant content items ---

export const chatCompletionResponseToContentItems = (
  responseBody: ?Object
): Array<ChatCompletionsResponseContentItem> => {
  if (!responseBody || !Array.isArray(responseBody.choices)) {
    throw new Error('Malformed Chat Completions response (no choices array)');
  }
  const choice = responseBody.choices[0];
  const message = choice && choice.message;
  if (!message) {
    throw new Error('Malformed Chat Completions response (no choice message)');
  }

  const items: Array<ChatCompletionsResponseContentItem> = [];
  if (typeof message.content === 'string' && message.content.length > 0) {
    items.push({
      type: 'output_text',
      status: 'completed',
      text: message.content,
      annotations: [],
    });
  }
  (message.tool_calls || []).forEach(toolCall => {
    items.push({
      type: 'function_call',
      status: 'completed',
      call_id: toolCall.id,
      name: toolCall.function.name,
      arguments: toolCall.function.arguments,
    });
  });
  return items;
};

// --- Transport ---

const MAX_ERROR_BODY_LENGTH = 500;

export const extractErrorMessage = (data: ?Object | string): string => {
  if (!data) return '';
  if (typeof data === 'string') {
    return data.length > MAX_ERROR_BODY_LENGTH
      ? `${data.slice(0, MAX_ERROR_BODY_LENGTH)}…`
      : data;
  }
  if (typeof data.error === 'string') return data.error;
  if (data.error && typeof data.error.message === 'string') {
    return data.error.message;
  }
  if (typeof data.message === 'string') return data.message;
  // FastAPI-style errors (io.net and vLLM stacks): {detail: "..."} or
  // {detail: [{msg: "..."}]}.
  if (typeof data.detail === 'string') return data.detail;
  if (Array.isArray(data.detail)) {
    const first = data.detail[0];
    if (first && typeof first.msg === 'string') return first.msg;
  }
  return '';
};

// Default transport: axios, resolving for every status so that the status
// handling stays in one place (sendChatCompletion).
export const axiosChatCompletionsTransport: ChatCompletionsTransport = async ({
  url,
  headers,
  body,
}) => {
  // $FlowFixMe[underconstrained-implicit-instantiation]
  const response = await axios.post(url, body, {
    headers,
    validateStatus: () => true,
    timeout: 120000,
  });
  return { status: response.status, data: response.data };
};

export const sendChatCompletion = async ({
  configuration,
  messages,
  tools,
  transport,
}: {|
  configuration: OpenAiCompatibleChatConfiguration,
  messages: Array<ChatCompletionsMessage>,
  tools?: ?Array<ChatCompletionsTool>,
  transport?: ?ChatCompletionsTransport,
|}): Promise<{|
  contentItems: Array<ChatCompletionsResponseContentItem>,
  usage: Object | null,
|}> => {
  const sendRequest = transport || axiosChatCompletionsTransport;
  const url = joinUrl(configuration.baseUrl, '/chat/completions');
  const body = {
    model: configuration.model,
    messages,
    // Some OpenAI-compatible servers (IO Intelligence included) default to
    // `tool_choice: "none"`, so an explicit "auto" is required whenever the
    // editor sends tools, otherwise its function-calling flows never trigger.
    ...(tools && tools.length > 0 ? { tools, tool_choice: 'auto' } : {}),
  };
  let response: ChatCompletionsTransportResponse;
  try {
    response = await sendRequest({
      url,
      headers: { Authorization: `Bearer ${configuration.apiKey}` },
      body,
    });
  } catch (error) {
    // Network-level failures (DNS, connection reset, timeout): surface the
    // endpoint so the user knows which provider configuration failed.
    throw new Error(
      `Chat Completions request to ${url} failed: ${
        error instanceof Error ? error.message : String(error)
      }`
    );
  }
  if (response.status < 200 || response.status >= 300) {
    const detail = extractErrorMessage(response.data);
    const suffix = detail ? `: ${detail}` : '';
    throw new Error(
      `Chat Completions request failed (${response.status})${suffix}`
    );
  }
  return {
    contentItems: chatCompletionResponseToContentItems(response.data),
    usage: response.data && response.data.usage ? response.data.usage : null,
  };
};
