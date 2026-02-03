/**
 * Message list component displaying chat history.
 */

import { useEffect, useRef } from 'react';
import { clsx } from 'clsx';
import { useChatStore } from '@/stores/chatStore';
import type { Message } from '@/types';

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === 'user';
  const isAssistant = message.role === 'assistant';

  // Check if content is an image URL
  const isImageUrl = message.content.startsWith('http') && 
    (message.content.includes('.png') || message.content.includes('.jpg') || message.content.includes('.jpeg'));

  return (
    <div
      className={clsx(
        'flex',
        isUser ? 'justify-end' : 'justify-start'
      )}
    >
      <div
        className={clsx(
          'max-w-[70%] rounded-2xl px-4 py-2',
          isUser
            ? 'bg-primary-500 text-white'
            : 'bg-white border border-gray-200 text-gray-900'
        )}
      >
        {isImageUrl ? (
          <div className="space-y-2">
            <img 
              src={message.content} 
              alt="Generated" 
              className="rounded-lg max-w-full h-auto"
              style={{ maxHeight: '300px' }}
            />
            <p className="text-xs opacity-70">Generated image</p>
          </div>
        ) : (
          <p className="whitespace-pre-wrap">{message.content}</p>
        )}
        
        {isAssistant && message.plan_json && (
          <div className="mt-2 pt-2 border-t border-gray-100">
            <span className="text-xs text-gray-500">
              {message.plan_json.steps?.length || 0} steps • {message.plan_json.status}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

export default function MessageList() {
  const { messages, isLoadingMessages } = useChatStore();
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  if (isLoadingMessages) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-gray-500">Loading messages...</div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-auto p-4">
      {messages.length === 0 ? (
        <div className="h-full flex items-center justify-center text-gray-500">
          <div className="text-center">
            <p className="text-lg mb-2">Start a conversation</p>
            <p className="text-sm">Type a message below to generate an image</p>
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          {messages.map((message) => (
            <MessageBubble key={message.id} message={message} />
          ))}
          <div ref={bottomRef} />
        </div>
      )}
    </div>
  );
}
