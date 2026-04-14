import { useState, useRef, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { ChatMessage } from './components/ChatMessage';
import { DebugSamplePreview } from './components/DebugSamplePreview';
import { DebugFixturePreview } from './components/DebugFixturePreview';
import { MessageInput } from './components/MessageInput';
import { Menu } from 'lucide-react';
import logo from '../assets/icon.svg';
import { nanoid } from 'nanoid';
import { parseStreaming } from './utils/parse-streaming';
import type { ConversationHistoryTurn, LlmRuntimeConfig, Source, Relate, SuggestedImage } from './utils/parse-streaming';

type UserLlmProvider = 'openai' | 'deepseek';

const OPENAI_MODELS = [
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

const DEEPSEEK_MODELS = ['deepseek-chat', 'deepseek-reasoner'] as const;

const getModelsForProvider = (provider: UserLlmProvider): readonly string[] =>
  provider === 'openai' ? OPENAI_MODELS : DEEPSEEK_MODELS;

const SERVER_MODEL_BY_PROVIDER: Record<UserLlmProvider, string> = {
  openai: 'gpt-5-mini',
  deepseek: 'deepseek-chat',
};

const SERVER_FIXED_MEMORY_TURNS = 3;
const MAX_USER_MEMORY_TURNS = 15;
const GREETING_MESSAGE_TEXT = "Hello! I'm ECEasy, your HKUST ECE assistant. How can I help you today?";

const API_KEY_PATTERN_BY_PROVIDER: Record<UserLlmProvider, RegExp> = {
  openai: /^sk-[A-Za-z0-9_-]{16,}$/,
  deepseek: /^sk-[A-Za-z0-9_-]{16,}$/,
};

const isStructurallyValidApiKey = (provider: UserLlmProvider, rawKey: string): boolean => {
  const key = rawKey.trim();
  if (!key) return false;
  return API_KEY_PATTERN_BY_PROVIDER[provider].test(key);
};

const isStructurallyValidBaseUrl = (rawUrl: string): boolean => {
  const value = rawUrl.trim();
  if (!value) return true;
  try {
    const parsed = new URL(value);
    return parsed.protocol === 'http:' || parsed.protocol === 'https:';
  } catch {
    return false;
  }
};

const DEFAULT_BASE_URL_BY_PROVIDER: Record<UserLlmProvider, string> = {
  openai: '',
  deepseek: '',
};

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  relates?: Relate[] | null;
  suggestedImages?: SuggestedImage[];
  isStreaming?: boolean;
}

interface ChatThread {
  id: string;
  title: string;
  updatedAt: number;
}

interface ChatState {
  threads: ChatThread[];
  messagesById: Record<string, Message[]>;
  currentChatId: string;
}

const CHAT_STATE_STORAGE_KEY = 'eceasy_chat_state_v1';

const createGreetingMessage = (): Message => ({
  id: nanoid(),
  role: 'assistant',
  content: GREETING_MESSAGE_TEXT,
});

const getChatTitle = (messages: Message[]): string => {
  const firstUser = messages.find((m) => m.role === 'user' && m.content.trim().length > 0);
  if (!firstUser) return 'New Chat session';
  const normalized = firstUser.content.replace(/\s+/g, ' ').trim();
  return normalized.length > 48 ? `${normalized.slice(0, 48)}...` : normalized;
};

const createThread = (id: string, messages: Message[], updatedAt = Date.now()): ChatThread => ({
  id,
  title: getChatTitle(messages),
  updatedAt,
});

const createDefaultChatState = (): ChatState => {
  const id = nanoid();
  const starter = [createGreetingMessage()];
  return {
    threads: [createThread(id, starter)],
    messagesById: { [id]: starter },
    currentChatId: id,
  };
};

const loadInitialChatState = (): ChatState => {
  if (typeof window === 'undefined') return createDefaultChatState();

  try {
    const raw = window.localStorage.getItem(CHAT_STATE_STORAGE_KEY);
    if (!raw) return createDefaultChatState();

    const parsed = JSON.parse(raw) as Omit<Partial<ChatState>, 'threads'> & {
      threads?: Array<Partial<ChatThread>>;
      messagesById?: Record<string, Message[]>;
      currentChatId?: string;
    };
    const messagesById = parsed.messagesById && typeof parsed.messagesById === 'object'
      ? parsed.messagesById
      : {};

    const rawThreads = Array.isArray(parsed.threads) ? parsed.threads : [];
    let threads: ChatThread[] = rawThreads
      .map((t) => {
        if (!t || typeof t.id !== 'string' || !t.id.trim()) return null;
        const id = t.id;
        return {
          id,
          title: typeof t.title === 'string' && t.title.trim() ? t.title : getChatTitle(messagesById[id] ?? []),
          updatedAt:
            typeof t.updatedAt === 'number' && Number.isFinite(t.updatedAt)
              ? t.updatedAt
              : Date.now(),
        };
      })
      .filter((t): t is ChatThread => t !== null);
    if (threads.length === 0) {
      const ids = Object.keys(messagesById);
      if (ids.length > 0) {
        threads = ids.map((id) => createThread(id, messagesById[id] ?? []));
      }
    }

    if (threads.length === 0) return createDefaultChatState();

    const currentChatId =
      parsed.currentChatId && threads.some((t) => t.id === parsed.currentChatId)
        ? parsed.currentChatId
        : threads[0].id;

    const normalizedMessagesById = { ...messagesById };
    for (const thread of threads) {
      if (!Array.isArray(normalizedMessagesById[thread.id]) || normalizedMessagesById[thread.id].length === 0) {
        normalizedMessagesById[thread.id] = [createGreetingMessage()];
      }
    }

    return {
      threads: threads.sort((a, b) => b.updatedAt - a.updatedAt),
      messagesById: normalizedMessagesById,
      currentChatId,
    };
  } catch {
    return createDefaultChatState();
  }
};

const syncThreadMeta = (threads: ChatThread[], chatId: string, messages: Message[]): ChatThread[] => {
  const updatedAt = Date.now();
  const title = getChatTitle(messages);
  const existingIndex = threads.findIndex((t) => t.id === chatId);

  let nextThreads: ChatThread[];
  if (existingIndex >= 0) {
    nextThreads = threads.map((t) => (t.id === chatId ? { ...t, title, updatedAt } : t));
  } else {
    nextThreads = [...threads, { id: chatId, title, updatedAt }];
  }

  return nextThreads.sort((a, b) => b.updatedAt - a.updatedAt);
};

function MainApp() {
  const [chatState, setChatState] = useState<ChatState>(() => loadInitialChatState());
  const [isLoading, setIsLoading] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [showLlmModal, setShowLlmModal] = useState(true);
  const [selectedProvider, setSelectedProvider] = useState<UserLlmProvider>('openai');
  const [selectedModel, setSelectedModel] = useState<string>('gpt-5-mini');
  const [userApiKey, setUserApiKey] = useState('');
  const [userBaseUrlByProvider, setUserBaseUrlByProvider] = useState<Record<UserLlmProvider, string>>(
    DEFAULT_BASE_URL_BY_PROVIDER,
  );
  const [useServerKey, setUseServerKey] = useState(false);
  const [llmConfigured, setLlmConfigured] = useState(false);
  const [llmConfigError, setLlmConfigError] = useState('');
  const [showApiKeyHint, setShowApiKeyHint] = useState(false);
  const [userMemoryTurns, setUserMemoryTurns] = useState<number>(SERVER_FIXED_MEMORY_TURNS);
  const [pendingMessage, setPendingMessage] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  // Keep a ref to the active AbortController so we can cancel if needed
  const abortRef = useRef<AbortController | null>(null);
  const activeStreamChatIdRef = useRef<string | null>(null);

  const { threads, currentChatId, messagesById } = chatState;
  const messages = messagesById[currentChatId] ?? [];
  const normalizedUserApiKey = userApiKey.trim();
  const isUserApiKeyStructurallyValid = isStructurallyValidApiKey(selectedProvider, normalizedUserApiKey);
  const normalizedUserBaseUrl = userBaseUrlByProvider[selectedProvider].trim();
  const isUserBaseUrlStructurallyValid = isStructurallyValidBaseUrl(normalizedUserBaseUrl);
  const serverFixedModel = SERVER_MODEL_BY_PROVIDER[selectedProvider];
  const effectiveMemoryTurns = useServerKey
    ? SERVER_FIXED_MEMORY_TURNS
    : Math.min(MAX_USER_MEMORY_TURNS, Math.max(0, userMemoryTurns));

  const updateChatMessages = (chatId: string, updater: (prev: Message[]) => Message[]) => {
    setChatState((prev) => {
      const current = prev.messagesById[chatId] ?? [createGreetingMessage()];
      const nextMessages = updater(current);
      return {
        ...prev,
        threads: syncThreadMeta(prev.threads, chatId, nextMessages),
        messagesById: {
          ...prev.messagesById,
          [chatId]: nextMessages,
        },
      };
    });
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    if (typeof window === 'undefined') return;
    window.localStorage.setItem(CHAT_STATE_STORAGE_KEY, JSON.stringify(chatState));
  }, [chatState]);

  // Mark the active streaming assistant reply as stopped.
  const stopActiveAssistantMessage = () => {
    const targetChatId = activeStreamChatIdRef.current ?? currentChatId;
    updateChatMessages(targetChatId, (prev) => {
      let targetId: string | null = null;
      for (let i = prev.length - 1; i >= 0; i--) {
        if (prev[i].role === 'assistant' && prev[i].isStreaming) {
          targetId = prev[i].id;
          break;
        }
      }
      if (!targetId) return prev;

      return prev.map((m) => {
        if (m.id !== targetId) return m;
        const alreadyMarked = m.content.includes('[Generation stopped]');
        const nextContent = m.content.trim().length
          ? alreadyMarked
            ? m.content
            : `${m.content}\n\n[Generation stopped]`
          : '[Generation stopped]';
        return { ...m, content: nextContent, isStreaming: false };
      });
    });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (!isUserApiKeyStructurallyValid) {
      setSelectedModel(serverFixedModel);
      return;
    }
    const models = getModelsForProvider(selectedProvider);
    setSelectedModel((prev) => (models.includes(prev) ? prev : models[0]));
  }, [selectedProvider, isUserApiKeyStructurallyValid, serverFixedModel]);

  const buildConversationHistory = (messages: Message[], memoryTurns: number): ConversationHistoryTurn[] => {
    const cleaned = messages
      .filter((m) => m.role === 'user' || m.role === 'assistant')
      .filter((m) => !m.isStreaming)
      .map((m) => ({ role: m.role, content: (m.content ?? '').trim() }))
      .filter((m) => m.content.length > 0)
      .filter((m) => !(m.role === 'assistant' && m.content === GREETING_MESSAGE_TEXT))
      .map((m) => ({ role: m.role, content: m.content.slice(0, 4000) }));

    const boundedTurns = Math.max(0, Math.min(MAX_USER_MEMORY_TURNS, memoryTurns));
    return cleaned.slice(-(boundedTurns * 2));
  };

  const runStreamingQuery = async (content: string) => {
    if (!content.trim() || isLoading) return;
    const targetChatId = currentChatId;
    const currentMessages = messagesById[targetChatId] ?? [];
    const conversationHistory = buildConversationHistory(currentMessages, effectiveMemoryTurns);
    activeStreamChatIdRef.current = targetChatId;

    // Cancel any in-flight request
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    // Add user message
    const userMessage: Message = {
      id: nanoid(),
      role: 'user',
      content,
    };

    // Add a placeholder assistant message that will be updated as tokens arrive
    const assistantId = nanoid();
    const assistantPlaceholder: Message = {
      id: assistantId,
      role: 'assistant',
      content: '',
      sources: [],
      relates: null,
      suggestedImages: [],
      isStreaming: true,
    };

    updateChatMessages(targetChatId, (prev) => [...prev, userMessage, assistantPlaceholder]);
    setIsLoading(true);

    const searchUuid = nanoid();

    try {
      const runtimeConfig: LlmRuntimeConfig = {
        llmProvider: selectedProvider,
        apiKey: useServerKey ? undefined : userApiKey.trim(),
        useServerKey,
        llmModel: useServerKey ? undefined : selectedModel,
        baseUrl: useServerKey ? undefined : normalizedUserBaseUrl || undefined,
        conversationHistory,
        memoryTurns: effectiveMemoryTurns,
      };

      await parseStreaming(
        controller,
        content,
        searchUuid,
        runtimeConfig,
        // onSources — called once the sources JSON is received
        (sources) => {
          updateChatMessages(targetChatId, (prev) =>
            prev.map((m) => (m.id === assistantId ? { ...m, sources } : m))
          );
        },
        // onMarkdown — called on every new chunk of LLM text
        (markdown) => {
          updateChatMessages(targetChatId, (prev) =>
            prev.map((m) => (m.id === assistantId ? { ...m, content: markdown } : m))
          );
          scrollToBottom();
        },
        // onRelates — called once the stream finishes
        (relates) => {
          updateChatMessages(targetChatId, (prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, relates, isStreaming: false } : m
            )
          );
          setIsLoading(false);
        },
        // onSuggestedImages — called once the stream finishes
        (suggestedImages) => {
          updateChatMessages(targetChatId, (prev) =>
            prev.map((m) => (m.id === assistantId ? { ...m, suggestedImages } : m))
          );
        },
        // onError
        (status) => {
          updateChatMessages(targetChatId, (prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? {
                    ...m,
                    content:
                      status === 429
                        ? 'Sorry, too many requests. Please try again later.'
                        : `Sorry, an error occurred (HTTP ${status}). Please try again.`,
                    isStreaming: false,
                  }
                : m
            )
          );
          setIsLoading(false);
        },
      );
    } catch (err: any) {
      if (err?.name === 'AbortError') {
        stopActiveAssistantMessage();
        setIsLoading(false);
      } else {
        updateChatMessages(targetChatId, (prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, content: 'Sorry, something went wrong. Please try again.', isStreaming: false }
              : m
          )
        );
        setIsLoading(false);
      }
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null;
      }
      activeStreamChatIdRef.current = null;
    }
  };

  const handleSendMessage = async (content: string) => {
    if (!llmConfigured) {
      setPendingMessage(content);
      setShowLlmModal(true);
      return;
    }
    await runStreamingQuery(content);
  };

  const confirmUseOwnKey = async () => {
    if (!normalizedUserApiKey) {
      setLlmConfigError('Please enter your API key, or choose "Use ECEasy key".');
      return;
    }
    if (!isUserApiKeyStructurallyValid) {
      setLlmConfigError(`Your ${selectedProvider} API key format looks invalid. Please check and try again.`);
      return;
    }
    if (!isUserBaseUrlStructurallyValid) {
      setLlmConfigError(`Your ${selectedProvider} base URL format looks invalid. Please enter a valid http(s) URL or leave it blank.`);
      return;
    }
    setUseServerKey(false);
    setLlmConfigured(true);
    setShowLlmModal(false);
    setLlmConfigError('');

    if (pendingMessage?.trim()) {
      const queued = pendingMessage;
      setPendingMessage(null);
      await runStreamingQuery(queued);
    }
  };

  const confirmUseServerKey = async () => {
    setUseServerKey(true);
    setSelectedModel(SERVER_MODEL_BY_PROVIDER[selectedProvider]);
    setLlmConfigured(true);
    setShowLlmModal(false);
    setLlmConfigError('');

    if (pendingMessage?.trim()) {
      const queued = pendingMessage;
      setPendingMessage(null);
      await runStreamingQuery(queued);
    }
  };

  const handleStopGeneration = () => {
    if (!isLoading) return;
    abortRef.current?.abort();
    stopActiveAssistantMessage();
    setIsLoading(false);
  };

  const handleNewChat = () => {
    abortRef.current?.abort();
    activeStreamChatIdRef.current = null;
    setIsLoading(false);
    const newChatId = nanoid();
    const starter = [createGreetingMessage()];
    setChatState((prev) => ({
      currentChatId: newChatId,
      messagesById: {
        ...prev.messagesById,
        [newChatId]: starter,
      },
      threads: syncThreadMeta(prev.threads, newChatId, starter),
    }));
  };

  const handleSelectChat = (chatId: string) => {
    setChatState((prev) => ({ ...prev, currentChatId: chatId }));
  };

  const handleDeleteChat = async (chatId: string) => {
    const chat = threads.find((t) => t.id === chatId);
    const chatTitle = chat?.title ?? 'Untitled Chat';
    if (!window.confirm(`Delete chat "${chatTitle}" permanently? This cannot be undone.`)) {
      return;
    }

    if (chatId === currentChatId && isLoading) {
      abortRef.current?.abort();
      activeStreamChatIdRef.current = null;
      setIsLoading(false);
    }

    // Best effort cleanup in backend cache/shelve; local UI state is source of truth.
    try {
      await fetch(`/api/chat/${encodeURIComponent(chatId)}`, { method: 'DELETE' });
    } catch {
      // Ignore network/backend delete failures and still remove local chat.
    }

    setChatState((prev) => {
      const nextMessagesById = { ...prev.messagesById };
      delete nextMessagesById[chatId];

      const nextThreads = prev.threads.filter((t) => t.id !== chatId);
      if (nextThreads.length === 0) {
        return createDefaultChatState();
      }

      const nextCurrentChatId =
        prev.currentChatId === chatId ? nextThreads[0].id : prev.currentChatId;

      return {
        ...prev,
        threads: nextThreads,
        messagesById: nextMessagesById,
        currentChatId: nextCurrentChatId,
      };
    });
  };

  return (
    <div className="flex h-screen bg-amber-50 text-gray-900 overflow-hidden">
      {/* Sidebar */}
      <Sidebar
        onNewChat={handleNewChat}
        chats={threads}
        currentChatId={currentChatId}
        onSelectChat={handleSelectChat}
        onDeleteChat={handleDeleteChat}
        isOpen={isSidebarOpen}
        setIsOpen={setIsSidebarOpen}
      />

      {/* Main Chat Area */}
      <div
        className={`flex-1 flex flex-col relative transition-all duration-300 ${
          isSidebarOpen ? 'ml-64' : 'ml-0'
        }`}
      >
        {/* Header */}
        <div className="border-b border-amber-200 bg-white/95 backdrop-blur-sm">
          <div className="w-full px-4 py-3 flex items-center justify-between">
            {/* Left: Toggle + Logo */}
            <div className="flex items-center gap-3">
              <button
                onClick={() => setIsSidebarOpen(!isSidebarOpen)}
                className="p-2 rounded-lg hover:bg-amber-50 transition-colors"
                title={isSidebarOpen ? 'Hide sidebar' : 'Show sidebar'}
              >
                <Menu size={20} />
              </button>
              <img src={logo} alt="ECEasy" className="h-12" />
            </div>

            {/* Centre: App Name */}
            <div className="absolute left-1/2 transform -translate-x-1/2">
              <h1 className="text-2xl font-bold">
                <span style={{ color: '#1e3a8a' }}>EC</span>
                <span style={{ color: '#3b82f6' }}>Easy</span>
              </h1>
            </div>

            {/* Right: balance spacer */}
            <div className="flex items-center gap-2">
              {llmConfigured && (
                <span
                  className={`text-xs px-2 py-1 rounded-full ${
                    useServerKey ? 'bg-amber-200 text-amber-900' : 'bg-blue-100 text-blue-900'
                  }`}
                >
                  {useServerKey
                    ? 'Using ECEasy key'
                    : `Using your ${selectedProvider} key (${selectedModel})`}
                </span>
              )}
              <button
                onClick={() => setShowLlmModal(true)}
                className="text-xs px-3 py-1.5 rounded-lg border border-amber-300 bg-white hover:bg-amber-50 transition-colors"
                title="LLM settings"
              >
                LLM Settings
              </button>
            </div>
          </div>
        </div>

        {/* Messages Container */}
        <div className="flex-1 overflow-y-auto bg-amber-50">
          <div className="min-h-full flex flex-col">
            {messages.map((message) => (
              <ChatMessage
                key={message.id}
                role={message.role}
                content={message.content}
                sources={message.sources}
                relates={message.relates}
                suggestedImages={message.suggestedImages}
                isStreaming={message.isStreaming}
                onRelatedQuestion={(q) => handleSendMessage(q)}
              />
            ))}

            {/* Initial loading indicator (before first token arrives) */}
            {isLoading && messages[messages.length - 1]?.content === '' && (
              <div className="flex gap-4 px-4 py-6 bg-amber-50">
                <div className="max-w-4xl mx-auto w-full flex gap-4">
                  <div className="flex-shrink-0">
                    <div className="w-8 h-8 rounded-full flex items-center justify-center bg-gradient-to-br from-amber-500 to-yellow-500">
                      <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
                    </div>
                  </div>
                  <div className="flex-1 min-w-0 pt-3">
                    <div className="flex gap-1">
                      <div className="w-2 h-2 bg-amber-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                      <div className="w-2 h-2 bg-amber-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                      <div className="w-2 h-2 bg-amber-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                    </div>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Area */}
        <MessageInput
          onSendMessage={handleSendMessage}
          disabled={isLoading || showLlmModal}
          isLoading={isLoading}
          onStopGeneration={handleStopGeneration}
        />
      </div>

      {showLlmModal && (
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
                  value={userBaseUrlByProvider[selectedProvider]}
                  onChange={(e) =>
                    setUserBaseUrlByProvider((prev) => ({
                      ...prev,
                      [selectedProvider]: e.target.value,
                    }))
                  }
                  placeholder={
                    selectedProvider === 'openai'
                      ? 'https://api.chatanywhere.org'
                      : 'https://api.deepseek.com'
                  }
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
                  onClick={() => {
                    setShowLlmModal(false);
                    setLlmConfigError('');
                  }}
                  className="px-4 py-2 rounded-lg border border-gray-300 text-sm hover:bg-gray-50"
                >
                  Cancel
                </button>
              )}
              <button
                onClick={confirmUseServerKey}
                className="px-4 py-2 rounded-lg border border-amber-300 text-sm text-amber-900 hover:bg-amber-50"
              >
                Skip - Use ECEasy key
              </button>
              <button
                onClick={confirmUseOwnKey}
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
      )}
    </div>
  );
}

export default function App() {
  const searchParams = typeof window !== 'undefined' ? new URLSearchParams(window.location.search) : null;
  const isDebugSampleMode = !!searchParams?.has('debugSample');
  const isDebugFixtureMode = !!searchParams?.has('debugFixture');

  if (isDebugSampleMode) return <DebugSamplePreview />;
  if (isDebugFixtureMode) return <DebugFixturePreview />;
  return <MainApp />;
}

