export type UserLlmProvider = 'openai' | 'deepseek';

export const OPENAI_MODELS = [
  'gpt-5.2',
  'gpt-5.1',
  'gpt-5',
  'gpt-4o',
  'gpt-4.1',
  'gpt-4o-mini',
  'gpt-3.5-turbo',
  'gpt-4.1-mini',
  'gpt-4.1-nano',
  'gpt-5-mini',
  'gpt-5-nano',
] as const;

export const DEEPSEEK_MODELS = ['deepseek-chat', 'deepseek-reasoner'] as const;

export const SERVER_MODEL_BY_PROVIDER: Record<UserLlmProvider, string> = {
  openai: 'gpt-5-mini',
  deepseek: 'deepseek-chat',
};

export const SERVER_FIXED_MEMORY_TURNS = 3;
export const MAX_USER_MEMORY_TURNS = 15;

export const API_KEY_PATTERN_BY_PROVIDER: Record<UserLlmProvider, RegExp> = {
  openai: /^sk-[A-Za-z0-9_-]{16,}$/,
  deepseek: /^sk-[A-Za-z0-9_-]{16,}$/,
};

export const DEFAULT_BASE_URL_BY_PROVIDER: Record<UserLlmProvider, string> = {
  openai: '',
  deepseek: '',
};

export const getModelsForProvider = (provider: UserLlmProvider): readonly string[] =>
  provider === 'openai' ? OPENAI_MODELS : DEEPSEEK_MODELS;

export const isStructurallyValidApiKey = (provider: UserLlmProvider, rawKey: string): boolean => {
  const key = rawKey.trim();
  if (!key) return false;
  return API_KEY_PATTERN_BY_PROVIDER[provider].test(key);
};

export const isStructurallyValidBaseUrl = (rawUrl: string): boolean => {
  const value = rawUrl.trim();
  if (!value) return true;
  try {
    const parsed = new URL(value);
    return parsed.protocol === 'http:' || parsed.protocol === 'https:';
  } catch {
    return false;
  }
};

