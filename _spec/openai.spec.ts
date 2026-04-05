import { OpenAIService, OpenAIMessage } from '../client';
import * as dotenv from 'dotenv';
dotenv.config();

describe('OpenAI Service', () => {
  let openaiService: OpenAIService;

  beforeEach(() => {
    console.log('OPENAI API Key:', process.env.OPENAI_API_KEY);
    openaiService = new OpenAIService(process.env.OPENAI_API_KEY || 'test-key');
    console.log('Test setup: OpenAI service initialized');
  });

  it('should create a chat completion', async () => {
    // Skip test if no API key is available
    if (!process.env.OPENAI_API_KEY) {
      console.log('Skipping test: No OpenAI API key available');
      return;
    }

    console.log('Running test: should create a chat completion');
    const messages: OpenAIMessage[] = [
      { role: 'system', content: 'You are a helpful assistant.' },
      { role: 'user', content: 'Tell me a short joke.' },
    ];

    console.log('Sending message to OpenAI:', messages);
    const result = await openaiService.createChatCompletion(messages);

    console.log('Received response from OpenAI:', result);
    expect(result.success).toBe(true);
    expect(result.content).toBeDefined();
    expect(typeof result.content).toBe('string');
    expect(result.content.length).toBeGreaterThan(0);
    console.log('Test passed: Chat completion created successfully');
  }, 30000); // Increased timeout for API call

  it('should handle errors gracefully', async () => {
    console.log('Running test: should handle errors gracefully');
    // Create a service with invalid API key
    const badService = new OpenAIService('invalid-key');

    const messages: OpenAIMessage[] = [{ role: 'user', content: 'Hello' }];

    console.log('Sending message with invalid API key');
    const result = await badService.createChatCompletion(messages);

    console.log('Received error response:', result);
    expect(result.success).toBe(false);
    expect(result.error).toBeDefined();
    console.log('Test passed: Error handled gracefully');
  });
});
