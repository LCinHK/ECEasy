import { nanoid } from 'nanoid';
import type { ChatState, ChatThread, Message } from '@/app/types/chat';

export const CHAT_STATE_STORAGE_KEY = 'eceasy_chat_state_v1';
export const GREETING_MESSAGE_TEXT = "Hello! I'm ECEasy, your HKUST ECE assistant. How can I help you today?";

export const createGreetingMessage = (): Message => ({
  id: nanoid(),
  role: 'assistant',
  content: GREETING_MESSAGE_TEXT,
});

export const getChatTitle = (messages: Message[]): string => {
  const firstUser = messages.find((m) => m.role === 'user' && m.content.trim().length > 0);
  if (!firstUser) return 'New Chat session';
  const normalized = firstUser.content.replace(/\s+/g, ' ').trim();
  return normalized.length > 48 ? `${normalized.slice(0, 48)}...` : normalized;
};

export const createThread = (id: string, messages: Message[], updatedAt = Date.now()): ChatThread => ({
  id,
  title: getChatTitle(messages),
  updatedAt,
});

export const createDefaultChatState = (): ChatState => {
  const id = nanoid();
  const starter = [createGreetingMessage()];
  return {
    threads: [createThread(id, starter)],
    messagesById: { [id]: starter },
    currentChatId: id,
  };
};

export const loadInitialChatState = (): ChatState => {
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

export const syncThreadMeta = (threads: ChatThread[], chatId: string, messages: Message[]): ChatThread[] => {
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

