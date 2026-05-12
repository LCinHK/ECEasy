import { useEffect, useRef, useState } from 'react';
import { Menu } from 'lucide-react';
import { nanoid } from 'nanoid';
import logo from '../assets/icon.svg';
import { Sidebar } from './components/Sidebar';
import { ChatMessage } from './components/ChatMessage';
import { DebugSamplePreview } from './components/DebugSamplePreview';
import { DebugFixturePreview } from './components/DebugFixturePreview';
import { MessageInput } from './components/MessageInput';
import { LlmSettingsModal } from './components/LlmSettingsModal';
import {
  CHAT_STATE_STORAGE_KEY,
  GREETING_MESSAGE_TEXT,
  createDefaultChatState,
  createGreetingMessage,
  loadInitialChatState,
  syncThreadMeta,
} from './utils/chat-state';
import { initMatomoTracking } from './utils/matomo';
import {
  DEFAULT_BASE_URL_BY_PROVIDER,
  MAX_USER_MEMORY_TURNS,
  SERVER_FIXED_MEMORY_TURNS,
  SERVER_MODEL_BY_PROVIDER,
  getModelsForProvider,
  isStructurallyValidApiKey,
  isStructurallyValidBaseUrl,
  type UserLlmProvider,
} from './utils/llm-config';
import { parseStreaming } from './utils/parse-streaming';
import type { ConversationHistoryTurn, LlmRuntimeConfig } from './utils/parse-streaming';
import type { ChatState, Message } from './types/chat';

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

  // Initialize Matomo once after app mount.
  useEffect(() => {
    initMatomoTracking();
  }, []);

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

  const buildConversationHistory = (messageList: Message[], memoryTurns: number): ConversationHistoryTurn[] => {
    const cleaned = messageList
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
            prev.map((m) => (m.id === assistantId ? { ...m, sources } : m)),
          );
        },
        // onMarkdown — called on every new chunk of LLM text
        (markdown) => {
          updateChatMessages(targetChatId, (prev) =>
            prev.map((m) => (m.id === assistantId ? { ...m, content: markdown } : m)),
          );
          scrollToBottom();
        },
        // onRelates — called once the stream finishes
        (relates) => {
          updateChatMessages(targetChatId, (prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, relates, isStreaming: false } : m,
            ),
          );
          setIsLoading(false);
        },
        // onSuggestedImages — called once the stream finishes
        (suggestedImages) => {
          updateChatMessages(targetChatId, (prev) =>
            prev.map((m) => (m.id === assistantId ? { ...m, suggestedImages } : m)),
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
                : m,
            ),
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
              : m,
          ),
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
      <Sidebar
        onNewChat={handleNewChat}
        chats={threads}
        currentChatId={currentChatId}
        onSelectChat={handleSelectChat}
        onDeleteChat={handleDeleteChat}
        isOpen={isSidebarOpen}
        setIsOpen={setIsSidebarOpen}
      />

      <div
        className={`flex-1 flex flex-col relative transition-all duration-300 ${
          isSidebarOpen ? 'ml-64' : 'ml-0'
        }`}
      >
        <div className="border-b border-amber-200 bg-white/95 backdrop-blur-sm">
          <div className="w-full px-4 py-3 flex items-center justify-between">
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

            <div className="flex items-center">
              <h1 className="text-2xl font-bold">
                <span style={{ color: '#1e3a8a' }}>EC</span>
                <span style={{ color: '#3b82f6' }}>Easy</span>
              </h1>
            </div>

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

        <MessageInput
          onSendMessage={handleSendMessage}
          disabled={isLoading || showLlmModal}
          isLoading={isLoading}
          onStopGeneration={handleStopGeneration}
        />
      </div>

      <LlmSettingsModal
        show={showLlmModal}
        llmConfigured={llmConfigured}
        selectedProvider={selectedProvider}
        setSelectedProvider={setSelectedProvider}
        userApiKey={userApiKey}
        setUserApiKey={setUserApiKey}
        showApiKeyHint={showApiKeyHint}
        setShowApiKeyHint={setShowApiKeyHint}
        userBaseUrl={userBaseUrlByProvider[selectedProvider]}
        setUserBaseUrl={(value) =>
          setUserBaseUrlByProvider((prev) => ({
            ...prev,
            [selectedProvider]: value,
          }))
        }
        isUserApiKeyStructurallyValid={isUserApiKeyStructurallyValid}
        isUserBaseUrlStructurallyValid={isUserBaseUrlStructurallyValid}
        selectedModel={selectedModel}
        setSelectedModel={setSelectedModel}
        serverFixedModel={serverFixedModel}
        userMemoryTurns={userMemoryTurns}
        setUserMemoryTurns={setUserMemoryTurns}
        llmConfigError={llmConfigError}
        onConfirmUseServerKey={confirmUseServerKey}
        onConfirmUseOwnKey={confirmUseOwnKey}
        onClose={() => {
          setShowLlmModal(false);
          setLlmConfigError('');
        }}
      />
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

