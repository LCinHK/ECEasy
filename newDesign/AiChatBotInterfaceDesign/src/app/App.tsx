import { useState, useRef, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { ChatMessage } from './components/ChatMessage';
import { MessageInput } from './components/MessageInput';
import { Menu } from 'lucide-react';
import logo from '../assets/icon.svg';
import { nanoid } from 'nanoid';
import { parseStreaming } from './utils/parse-streaming';
import type { Source, Relate } from './utils/parse-streaming';

type UserLlmProvider = 'openai' | 'deepseek';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  relates?: Relate[] | null;
  isStreaming?: boolean;
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content: "Hello! I'm ECEasy, your HKUST ECE assistant. How can I help you today?",
    },
  ]);
  const [currentChatId, setCurrentChatId] = useState('1');
  const [isLoading, setIsLoading] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [showLlmModal, setShowLlmModal] = useState(true);
  const [selectedProvider, setSelectedProvider] = useState<UserLlmProvider>('openai');
  const [userApiKey, setUserApiKey] = useState('');
  const [useServerKey, setUseServerKey] = useState(false);
  const [llmConfigured, setLlmConfigured] = useState(false);
  const [llmConfigError, setLlmConfigError] = useState('');
  const [pendingMessage, setPendingMessage] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  // Keep a ref to the active AbortController so we can cancel if needed
  const abortRef = useRef<AbortController | null>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Mark the active streaming assistant reply as stopped.
  const stopActiveAssistantMessage = () => {
    setMessages((prev) => {
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

  const runStreamingQuery = async (content: string) => {
    if (!content.trim() || isLoading) return;

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
      isStreaming: true,
    };

    setMessages((prev) => [...prev, userMessage, assistantPlaceholder]);
    setIsLoading(true);

    const searchUuid = nanoid();

    try {
      await parseStreaming(
        controller,
        content,
        searchUuid,
        {
          llmProvider: selectedProvider,
          apiKey: useServerKey ? undefined : userApiKey.trim(),
          useServerKey,
        },
        // onSources — called once the sources JSON is received
        (sources) => {
          setMessages((prev) =>
            prev.map((m) => (m.id === assistantId ? { ...m, sources } : m))
          );
        },
        // onMarkdown — called on every new chunk of LLM text
        (markdown) => {
          setMessages((prev) =>
            prev.map((m) => (m.id === assistantId ? { ...m, content: markdown } : m))
          );
          scrollToBottom();
        },
        // onRelates — called once the stream finishes
        (relates) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, relates, isStreaming: false } : m
            )
          );
          setIsLoading(false);
        },
        // onError
        (status) => {
          setMessages((prev) =>
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
        setMessages((prev) =>
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
    if (!userApiKey.trim()) {
      setLlmConfigError('Please enter your API key, or choose "Use ECEasy key".');
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
    setIsLoading(false);
    setMessages([
      {
        id: nanoid(),
        role: 'assistant',
        content: "Hello! I'm ECEasy, your HKUST ECE assistant. How can I help you today?",
      },
    ]);
    setCurrentChatId(nanoid());
  };

  const handleSelectChat = (chatId: string) => {
    setCurrentChatId(chatId);
    // Sidebar history is a future feature — for now just acknowledge selection
  };

  return (
    <div className="flex h-screen bg-amber-50 text-gray-900 overflow-hidden">
      {/* Sidebar */}
      <Sidebar
        onNewChat={handleNewChat}
        currentChatId={currentChatId}
        onSelectChat={handleSelectChat}
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
                  {useServerKey ? 'Using ECEasy key' : `Using your ${selectedProvider} key`}
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
                <label className="block text-sm font-medium text-gray-700 mb-1">Your API key</label>
                <input
                  type="password"
                  value={userApiKey}
                  onChange={(e) => setUserApiKey(e.target.value)}
                  placeholder={selectedProvider === 'openai' ? 'sk-...' : 'sk-...'}
                  className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-amber-400"
                />
                <p className="mt-1 text-xs text-gray-500">
                  Key is sent only with chat requests and is not stored in this UI.
                </p>
              </div>

              <div className="rounded-lg bg-amber-50 border border-amber-200 p-3 text-sm text-amber-900">
                If you skip and use ECEasy&apos;s key, requests may incur our API costs and might be rate-limited.
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
                className="px-4 py-2 rounded-lg bg-amber-600 text-white text-sm hover:bg-amber-700"
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

