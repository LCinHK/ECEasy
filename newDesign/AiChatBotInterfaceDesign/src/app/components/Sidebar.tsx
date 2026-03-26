import { Plus, MessageSquare } from 'lucide-react';

interface Chat {
  id: string;
  title: string;
  updatedAt: number;
}

interface SidebarProps {
  onNewChat: () => void;
  chats: Chat[];
  currentChatId: string;
  onSelectChat: (chatId: string) => void;
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
}

const formatChatTimestamp = (updatedAt: number): string => {
  const date = new Date(updatedAt);
  if (Number.isNaN(date.getTime())) return '';

  const now = new Date();
  const isToday =
    date.getFullYear() === now.getFullYear() &&
    date.getMonth() === now.getMonth() &&
    date.getDate() === now.getDate();

  if (isToday) return 'Today';

  const yesterday = new Date(now);
  yesterday.setDate(now.getDate() - 1);
  const isYesterday =
    date.getFullYear() === yesterday.getFullYear() &&
    date.getMonth() === yesterday.getMonth() &&
    date.getDate() === yesterday.getDate();

  if (isYesterday) return 'Yesterday';

  return date.toLocaleDateString();
};

export function Sidebar({ onNewChat, chats, currentChatId, onSelectChat, isOpen, setIsOpen }: SidebarProps) {

  return (
    <>
      {/* Sidebar */}
      <aside
        className={`fixed inset-y-0 left-0 z-40 w-64 bg-amber-50 border-r border-amber-200 flex flex-col transition-transform duration-300 ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {/* New Chat Button */}
        <div className="p-3 border-b border-amber-200">
          <button
            onClick={onNewChat}
            className="w-full flex items-center gap-3 px-4 py-3 rounded-lg bg-amber-100 hover:bg-amber-200 transition-colors text-gray-900"
          >
            <Plus size={18} />
            <span>New chat</span>
          </button>
        </div>

        {/* Chat History */}
        <div className="flex-1 overflow-y-auto p-3 space-y-1">
          {chats.map((chat) => (
            <button
              key={chat.id}
              onClick={() => {
                onSelectChat(chat.id);
                if (window.innerWidth < 1024) setIsOpen(false);
              }}
              className={`w-full flex items-start gap-3 px-3 py-3 rounded-lg transition-colors text-left ${
                currentChatId === chat.id
                  ? 'bg-amber-100 text-gray-900'
                  : 'text-gray-600 hover:bg-amber-100/50 hover:text-gray-900'
              }`}
            >
              <MessageSquare size={18} className="mt-0.5 flex-shrink-0" />
              <div className="flex-1 min-w-0">
                <div className="truncate text-sm">{chat.title}</div>
                <div className="text-xs text-gray-400 mt-0.5">{formatChatTimestamp(chat.updatedAt)}</div>
              </div>
            </button>
          ))}
        </div>

        {/* User Section */}
        <div className="p-4 border-t border-amber-200">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-amber-500 to-orange-500 flex items-center justify-center text-sm text-white">
              U
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-sm truncate text-gray-900">Demo Account</div>
            </div>
          </div>
        </div>
      </aside>

      {/* Overlay for mobile */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-30 lg:hidden"
          onClick={() => setIsOpen(false)}
        />
      )}
    </>
  );
}