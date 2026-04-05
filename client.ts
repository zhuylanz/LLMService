export * from './types';
export { LLMService } from './service';

import { LLMService } from './service';
import type { LLMServiceOptions } from './types';

export class OpenAIService extends LLMService {
  constructor(apiKey: string, organization?: string) {
    super({ provider: 'openai', apiKey, organization });
  }
}

export class ClaudeService extends LLMService {
  constructor(apiKey: string, baseURL?: string) {
    super({ provider: 'claude', apiKey, baseURL });
  }
}

export class GeminiService extends LLMService {
  constructor(apiKey: string, baseURL?: string) {
    super({ provider: 'gemini', apiKey, baseURL });
  }
}

export const createLLMService = (
  configOrApiKey: string | LLMServiceOptions,
  organization?: string,
): LLMService => {
  return new LLMService(configOrApiKey, organization);
};

export const createOpenAIService = (
  apiKey: string,
  organization?: string,
): OpenAIService => {
  return new OpenAIService(apiKey, organization);
};

export const createClaudeService = (
  apiKey: string,
  baseURL?: string,
): ClaudeService => {
  return new ClaudeService(apiKey, baseURL);
};

export const createGeminiService = (
  apiKey: string,
  baseURL?: string,
): GeminiService => {
  return new GeminiService(apiKey, baseURL);
};

export default {
  LLMService,
  createLLMService,
  OpenAIService,
  createOpenAIService,
  ClaudeService,
  createClaudeService,
  GeminiService,
  createGeminiService,
};
