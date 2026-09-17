// @flow
import {
  joinUrl,
  translateAiRequestMessagesToChatMessages,
  chatCompletionResponseToContentItems,
  sendChatCompletion,
  extractErrorMessage,
  IO_INTELLIGENCE_DEFAULTS,
} from './OpenAiCompatibleChatClient';
import type {
  AiRequestUserMessage,
  AiRequestAssistantMessage,
  AiRequestFunctionCallOutput,
} from '../Utils/GDevelopServices/Generation';

const makeTransport = (status: number, data: Object) =>
  jest.fn(async (request: any) => ({ status, data }));

const userMessage = (texts: Array<string>): AiRequestUserMessage => ({
  type: 'message',
  status: 'completed',
  role: 'user',
  content: texts.map(text => ({
    type: 'user_request',
    status: 'completed',
    text,
  })),
});

const assistantMessage = (content: Array<any>): AiRequestAssistantMessage => ({
  type: 'message',
  status: 'completed',
  role: 'assistant',
  content,
});

const textContent = (text: string): any => ({
  type: 'output_text',
  status: 'completed',
  text,
  annotations: [],
});

const functionCallContent = (
  callId: string,
  name: string,
  args: string
): any => ({
  type: 'function_call',
  status: 'completed',
  call_id: callId,
  name,
  arguments: args,
});

const functionCallOutput = (
  callId: string,
  output: string
): AiRequestFunctionCallOutput => ({
  type: 'function_call_output',
  call_id: callId,
  output,
});

describe('OpenAiCompatibleChatClient', () => {
  describe('joinUrl', () => {
    it('joins a base URL without trailing slash', () => {
      expect(joinUrl('https://example.com/v1', '/chat/completions')).toBe(
        'https://example.com/v1/chat/completions'
      );
    });

    it('strips trailing slashes from the base URL', () => {
      expect(joinUrl('https://example.com/v1///', '/chat/completions')).toBe(
        'https://example.com/v1/chat/completions'
      );
    });
  });

  describe('translateAiRequestMessagesToChatMessages', () => {
    it('translates a user message with a single part', () => {
      expect(
        translateAiRequestMessagesToChatMessages([userMessage(['Hello'])])
      ).toEqual([{ role: 'user', content: 'Hello' }]);
    });

    it('joins multiple user_request parts', () => {
      expect(
        translateAiRequestMessagesToChatMessages([
          userMessage(['First', 'Second']),
        ])
      ).toEqual([{ role: 'user', content: 'First\n\nSecond' }]);
    });

    it('translates an assistant text message', () => {
      expect(
        translateAiRequestMessagesToChatMessages([
          assistantMessage([textContent('Hi there')]),
        ])
      ).toEqual([{ role: 'assistant', content: 'Hi there' }]);
    });

    it('translates assistant function calls into tool_calls', () => {
      expect(
        translateAiRequestMessagesToChatMessages([
          assistantMessage([
            textContent('Calling a tool'),
            functionCallContent('call-1', 'generate_events', '{"a":1}'),
          ]),
        ])
      ).toEqual([
        {
          role: 'assistant',
          content: 'Calling a tool',
          tool_calls: [
            {
              id: 'call-1',
              type: 'function',
              function: { name: 'generate_events', arguments: '{"a":1}' },
            },
          ],
        },
      ]);
    });

    it('gives assistant messages without text a null content', () => {
      expect(
        translateAiRequestMessagesToChatMessages([
          assistantMessage([functionCallContent('call-1', 'fn', '{}')]),
        ])
      ).toEqual([
        {
          role: 'assistant',
          content: null,
          tool_calls: [
            {
              id: 'call-1',
              type: 'function',
              function: { name: 'fn', arguments: '{}' },
            },
          ],
        },
      ]);
    });

    it('drops reasoning content', () => {
      expect(
        translateAiRequestMessagesToChatMessages([
          assistantMessage([
            {
              type: 'reasoning',
              status: 'completed',
              summary: { text: 'thinking', type: 'summary_text' },
            },
            textContent('Answer'),
          ]),
        ])
      ).toEqual([{ role: 'assistant', content: 'Answer' }]);
    });

    it('translates function_call_output records into tool messages', () => {
      expect(
        translateAiRequestMessagesToChatMessages([
          functionCallOutput('call-1', '{"ok":true}'),
        ])
      ).toEqual([
        { role: 'tool', content: '{"ok":true}', tool_call_id: 'call-1' },
      ]);
    });

    it('preserves the order of a mixed conversation', () => {
      const chatMessages = translateAiRequestMessagesToChatMessages([
        userMessage(['Question']),
        assistantMessage([functionCallContent('call-1', 'fn', '{}')]),
        functionCallOutput('call-1', 'result'),
        assistantMessage([textContent('Done')]),
      ]);
      expect(chatMessages.map(message => message.role)).toEqual([
        'user',
        'assistant',
        'tool',
        'assistant',
      ]);
    });
  });

  describe('chatCompletionResponseToContentItems', () => {
    it('maps a text answer to an output_text item', () => {
      expect(
        chatCompletionResponseToContentItems({
          choices: [{ message: { role: 'assistant', content: 'Answer' } }],
        })
      ).toEqual([textContent('Answer')]);
    });

    it('maps tool_calls to function_call items', () => {
      expect(
        chatCompletionResponseToContentItems({
          choices: [
            {
              message: {
                role: 'assistant',
                content: null,
                tool_calls: [
                  {
                    id: 'call-9',
                    type: 'function',
                    function: { name: 'fn', arguments: '{"x":2}' },
                  },
                ],
              },
            },
          ],
        })
      ).toEqual([functionCallContent('call-9', 'fn', '{"x":2}')]);
    });

    it('emits no text item for an empty content', () => {
      expect(
        chatCompletionResponseToContentItems({
          choices: [{ message: { role: 'assistant', content: '' } }],
        })
      ).toEqual([]);
    });

    it('throws on a malformed response', () => {
      expect(() => chatCompletionResponseToContentItems(null)).toThrow(
        /Malformed/
      );
      expect(() =>
        chatCompletionResponseToContentItems({ choices: [] })
      ).toThrow(/Malformed/);
    });
  });

  describe('extractErrorMessage', () => {
    it('reads nested error messages', () => {
      expect(extractErrorMessage({ error: { message: 'bad key' } })).toBe(
        'bad key'
      );
      expect(extractErrorMessage({ error: 'nope' })).toBe('nope');
      expect(extractErrorMessage({ message: 'nope' })).toBe('nope');
      expect(extractErrorMessage('plain text')).toBe('plain text');
      expect(extractErrorMessage(null)).toBe('');
    });
  });

  describe('sendChatCompletion', () => {
    const configuration = {
      baseUrl: 'https://api.example.com/v1/',
      apiKey: 'test-key',
      model: 'org/model-name',
    };

    it('sends a Bearer-authenticated request to the joined URL', async () => {
      const transport = makeTransport(200, {
        choices: [{ message: { role: 'assistant', content: 'ok' } }],
        usage: { total_tokens: 7 },
      });
      const result = await sendChatCompletion({
        configuration,
        messages: [{ role: 'user', content: 'hi' }],
        transport,
      });
      const request = transport.mock.calls[0][0];
      expect(request.url).toBe('https://api.example.com/v1/chat/completions');
      expect(request.headers.Authorization).toBe('Bearer test-key');
      expect(request.body.model).toBe('org/model-name');
      expect(request.body.messages).toEqual([{ role: 'user', content: 'hi' }]);
      expect(request.body.tools).toBeUndefined();
      expect(request.body.tool_choice).toBeUndefined();
      expect(result.contentItems).toEqual([textContent('ok')]);
      expect(result.usage).toEqual({ total_tokens: 7 });
    });

    it('includes tools and tool_choice "auto" when tools are given', async () => {
      const transport = makeTransport(200, {
        choices: [{ message: { role: 'assistant', content: 'ok' } }],
      });
      await sendChatCompletion({
        configuration,
        messages: [{ role: 'user', content: 'hi' }],
        tools: [
          {
            type: 'function',
            function: { name: 'fn', parameters: { type: 'object' } },
          },
        ],
        transport,
      });
      const request = transport.mock.calls[0][0];
      expect(request.body.tools).toEqual([
        {
          type: 'function',
          function: { name: 'fn', parameters: { type: 'object' } },
        },
      ]);
      expect(request.body.tool_choice).toBe('auto');
    });

    it('throws with the status and server message on failure', async () => {
      const transport = makeTransport(401, {
        error: { message: 'Invalid API key' },
      });
      await expect(
        sendChatCompletion({
          configuration,
          messages: [{ role: 'user', content: 'hi' }],
          transport,
        })
      ).rejects.toThrow(
        'Chat Completions request failed (401): Invalid API key'
      );
    });

    it('exposes the io.net defaults', () => {
      expect(IO_INTELLIGENCE_DEFAULTS.baseUrl).toBe(
        'https://api.intelligence.io.solutions/api/v1'
      );
      expect(IO_INTELLIGENCE_DEFAULTS.model).toBe(
        'meta-llama/Llama-3.3-70B-Instruct'
      );
    });
  });
});
