import type { UserLlmProvider } from '@/app/utils/llm-config';
import {
  MAX_USER_MEMORY_TURNS,
  SERVER_FIXED_MEMORY_TURNS,
  getModelsForProvider,
} from '@/app/utils/llm-config';

interface LlmSettingsModalProps {
  show: boolean;
  llmConfigured: boolean;
  selectedProvider: UserLlmProvider;
  setSelectedProvider: (provider: UserLlmProvider) => void;
  userApiKey: string;
  setUserApiKey: (value: string) => void;
  showApiKeyHint: boolean;
  setShowApiKeyHint: (updater: (prev: boolean) => boolean) => void;
  userBaseUrl: string;
  setUserBaseUrl: (value: string) => void;
  isUserApiKeyStructurallyValid: boolean;
  isUserBaseUrlStructurallyValid: boolean;
  selectedModel: string;
  setSelectedModel: (value: string) => void;
  serverFixedModel: string;
  userMemoryTurns: number;
  setUserMemoryTurns: (value: number) => void;
  llmConfigError: string;
  onConfirmUseServerKey: () => void;
  onConfirmUseOwnKey: () => void;
  onClose: () => void;
}

export function LlmSettingsModal({
  show,
  llmConfigured,
  selectedProvider,
  setSelectedProvider,
  userApiKey,
  setUserApiKey,
  showApiKeyHint,
  setShowApiKeyHint,
  userBaseUrl,
  setUserBaseUrl,
  isUserApiKeyStructurallyValid,
  isUserBaseUrlStructurallyValid,
  selectedModel,
  setSelectedModel,
  serverFixedModel,
  userMemoryTurns,
  setUserMemoryTurns,
  llmConfigError,
  onConfirmUseServerKey,
  onConfirmUseOwnKey,
  onClose,
}: LlmSettingsModalProps) {
  if (!show) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/35 p-4">
      <div className="w-full max-w-lg rounded-2xl bg-white p-6 shadow-2xl border border-amber-200">
        <h2 className="text-xl font-bold text-gray-900">Choose your LLM access</h2>
        <p className="mt-2 text-sm text-gray-600">
          Use your own API key (recommended) or continue with ECEasy&apos;s shared key.
        </p>

        <div className="mt-5 space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Provider</label>
            <select
              value={selectedProvider}
              onChange={(e) => setSelectedProvider(e.target.value as UserLlmProvider)}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-amber-400"
            >
              <option value="openai">OpenAI</option>
              <option value="deepseek">DeepSeek</option>
            </select>
          </div>

          <div>
            <div className="mb-1 flex items-center gap-2">
              <label className="block text-sm font-medium text-gray-700">Your API key</label>
              <button
                type="button"
                onClick={() => setShowApiKeyHint((prev) => !prev)}
                className="text-xs px-1.5 py-0.5 rounded border border-amber-300 text-amber-800 hover:bg-amber-50"
                aria-label="What is an API key?"
                title="What is an API key?"
              >
                ?
              </button>
            </div>
            <input
              type="password"
              value={userApiKey}
              onChange={(e) => setUserApiKey(e.target.value)}
              placeholder={selectedProvider === 'openai' ? 'sk-...' : 'sk-...'}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-amber-400"
            />
            {showApiKeyHint && (
              <p className="mt-2 text-xs text-gray-600">
                An API key is a private token that lets apps use an AI provider on your account.
                You can get a free compatible key from{' '}
                <a
                  href="https://github.com/chatanywhere/GPT_API_free"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 underline hover:text-blue-800"
                >
                  chatanywhere/GPT_API_free
                </a>
                .
              </p>
            )}
            <p className="mt-1 text-xs text-gray-500">
              Key is sent only with chat requests and is not stored currently.
            </p>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Base URL (optional)</label>
            <input
              type="url"
              value={userBaseUrl}
              onChange={(e) => setUserBaseUrl(e.target.value)}
              placeholder={selectedProvider === 'openai' ? 'https://api.chatanywhere.org' : 'https://api.deepseek.com'}
              disabled={!isUserApiKeyStructurallyValid}
              className={`w-full rounded-lg border px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-amber-400 ${
                !isUserApiKeyStructurallyValid
                  ? 'border-gray-200 bg-gray-100 text-gray-400 cursor-not-allowed'
                  : isUserBaseUrlStructurallyValid
                    ? 'border-gray-300 bg-white text-gray-900'
                    : 'border-red-300 bg-white text-gray-900'
              }`}
            />
            <p className={`mt-1 text-xs ${isUserBaseUrlStructurallyValid ? 'text-gray-500' : 'text-red-600'}`}>
              {!isUserApiKeyStructurallyValid
                ? 'Enter a valid API key first to unlock endpoint selection.'
                : 'Leave blank to use ECEasy defaults. Enter a valid http(s) endpoint if you want to override the provider URL.'}
            </p>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Model</label>
            <select
              value={isUserApiKeyStructurallyValid ? selectedModel : serverFixedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              disabled={!isUserApiKeyStructurallyValid}
              className={`w-full rounded-lg border px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-amber-400 ${
                isUserApiKeyStructurallyValid
                  ? 'border-gray-300 bg-white text-gray-900'
                  : 'border-gray-200 bg-gray-100 text-gray-400 cursor-not-allowed'
              }`}
            >
              {isUserApiKeyStructurallyValid
                ? getModelsForProvider(selectedProvider).map((model) => (
                    <option key={model} value={model}>
                      {model}
                    </option>
                  ))
                : (
                    <option value={serverFixedModel}>{serverFixedModel}</option>
                  )}
            </select>
            <p className="mt-1 text-xs text-gray-500">
              {isUserApiKeyStructurallyValid
                ? 'This selected model is used only when you choose "Use my key".'
                : `Using fixed server model: ${serverFixedModel}. Enter a valid ${selectedProvider} API key (starts with sk-) to unlock model selection.`}
            </p>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Memory window (turns)</label>
            {isUserApiKeyStructurallyValid ? (
              <>
                <input
                  type="range"
                  min={0}
                  max={MAX_USER_MEMORY_TURNS}
                  step={1}
                  value={userMemoryTurns}
                  onChange={(e) => setUserMemoryTurns(Number(e.target.value))}
                  className="w-full accent-amber-600"
                />
                <div className="mt-1 text-xs text-gray-600">
                  {userMemoryTurns} turn{userMemoryTurns === 1 ? '' : 's'} (0 = stateless)
                </div>
              </>
            ) : (
              <>
                <input
                  type="range"
                  min={SERVER_FIXED_MEMORY_TURNS}
                  max={SERVER_FIXED_MEMORY_TURNS}
                  value={SERVER_FIXED_MEMORY_TURNS}
                  disabled
                  className="w-full accent-gray-300 cursor-not-allowed"
                />
                <div className="mt-1 text-xs text-gray-500">
                  Using fixed server memory: last {SERVER_FIXED_MEMORY_TURNS} turns. The shared key may trim history further to stay under its ~4096-token free limit. Enter a valid key to customize 0-{MAX_USER_MEMORY_TURNS}.
                </div>
              </>
            )}
          </div>

          <div className="rounded-lg bg-amber-50 border border-amber-200 p-3 text-sm text-amber-900">
            If you skip and use ECEasy&apos;s key, requests may incur our API costs and might be rate-limited (May not be available at all times).
          </div>

          {llmConfigError && <p className="text-sm text-red-600">{llmConfigError}</p>}
        </div>

        <div className="mt-6 flex flex-wrap gap-2 justify-end">
          {llmConfigured && (
            <button
              onClick={onClose}
              className="px-4 py-2 rounded-lg border border-gray-300 text-sm hover:bg-gray-50"
            >
              Cancel
            </button>
          )}
          <button
            onClick={onConfirmUseServerKey}
            className="px-4 py-2 rounded-lg border border-amber-300 text-sm text-amber-900 hover:bg-amber-50"
          >
            Skip - Use ECEasy key
          </button>
          <button
            onClick={onConfirmUseOwnKey}
            disabled={!isUserApiKeyStructurallyValid || !isUserBaseUrlStructurallyValid}
            className={`px-4 py-2 rounded-lg text-white text-sm ${
              isUserApiKeyStructurallyValid && isUserBaseUrlStructurallyValid
                ? 'bg-amber-600 hover:bg-amber-700'
                : 'bg-amber-300 cursor-not-allowed'
            }`}
          >
            Use my key
          </button>
        </div>
      </div>
    </div>
  );
}

