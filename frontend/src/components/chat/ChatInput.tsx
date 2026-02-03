/**
 * Chat input component for sending messages.
 */

import { useState, KeyboardEvent } from 'react';
import { useChatStore } from '@/stores/chatStore';

export default function ChatInput() {
  const [input, setInput] = useState('');
  const { currentSession, isPlanning, sendMessageStream } = useChatStore();

  const handleSend = () => {
    if (!input.trim() || !currentSession || isPlanning) return;
    
    sendMessageStream(input.trim());
    setInput('');
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="border-t border-gray-200 bg-white p-4">
      <div className="flex items-end gap-3">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            currentSession
              ? 'Describe the image you want to generate...'
              : 'Select or create a session first'
          }
          disabled={!currentSession || isPlanning}
          className="flex-1 resize-none rounded-lg border border-gray-300 px-4 py-3 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed"
          rows={2}
        />
        <button
          onClick={handleSend}
          disabled={!input.trim() || !currentSession || isPlanning}
          className="px-6 py-3 bg-primary-500 text-white rounded-lg hover:bg-primary-600 transition-colors disabled:bg-gray-300 disabled:cursor-not-allowed font-medium"
        >
          {isPlanning ? 'Generating...' : 'Send'}
        </button>
      </div>
      <p className="text-xs text-gray-500 mt-2">
        Press Enter to send, Shift+Enter for new line
      </p>
    </div>
  );
}
