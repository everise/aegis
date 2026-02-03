/**
 * Session list sidebar component.
 */

import { useNavigate } from 'react-router-dom';
import { clsx } from 'clsx';
import { useChatStore } from '@/stores/chatStore';

export default function SessionList() {
  const navigate = useNavigate();
  const { sessions, currentSession, createSession, deleteSession } = useChatStore();

  const handleCreateSession = async () => {
    const session = await createSession();
    navigate(`/chat/${session.id}`);
  };

  const handleSelectSession = (sessionId: number) => {
    navigate(`/chat/${sessionId}`);
  };

  const handleDeleteSession = async (e: React.MouseEvent, sessionId: number) => {
    e.stopPropagation();
    if (confirm('Delete this session?')) {
      await deleteSession(sessionId);
      if (currentSession?.id === sessionId) {
        navigate('/chat');
      }
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="p-3 border-b border-gray-200">
        <button
          onClick={handleCreateSession}
          className="w-full px-4 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600 transition-colors font-medium"
        >
          + New Session
        </button>
      </div>

      <div className="flex-1 overflow-auto p-2">
        {sessions.length === 0 ? (
          <p className="text-center text-gray-500 text-sm py-4">
            No sessions yet
          </p>
        ) : (
          <div className="space-y-1">
            {sessions.map((session) => (
              <div
                key={session.id}
                onClick={() => handleSelectSession(session.id)}
                className={clsx(
                  'p-3 rounded-lg cursor-pointer transition-colors group',
                  currentSession?.id === session.id
                    ? 'bg-primary-100 border border-primary-200'
                    : 'hover:bg-gray-100'
                )}
              >
                <div className="flex items-center justify-between">
                  <span className="font-medium text-sm text-gray-900">
                    Session #{session.id}
                  </span>
                  <button
                    onClick={(e) => handleDeleteSession(e, session.id)}
                    className="opacity-0 group-hover:opacity-100 text-gray-400 hover:text-red-500 transition-opacity"
                  >
                    ×
                  </button>
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  {session.task_type} • {session.message_count} messages
                </div>
                <div className="text-xs text-gray-400 mt-1">
                  {new Date(session.created_at).toLocaleDateString()}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
