import type { Relate, Source, SuggestedImage } from '@/app/utils/parse-streaming';

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  relates?: Relate[] | null;
  suggestedImages?: SuggestedImage[];
  isStreaming?: boolean;
}

export interface ChatThread {
  id: string;
  title: string;
  updatedAt: number;
}

export interface ChatState {
  threads: ChatThread[];
  messagesById: Record<string, Message[]>;
  currentChatId: string;
}

