/**
 * Chat page component.
 */

import { useEffect } from 'react';
import { useParams } from 'react-router-dom';
import SessionList from '@/components/chat/SessionList';
import MessageList from '@/components/chat/MessageList';
import ChatInput from '@/components/chat/ChatInput';
import PlanViewer from '@/components/plan/PlanViewer';
import { useChatStore } from '@/stores/chatStore';

export default function ChatPage() {
  const { sessionId } = useParams();
  const { 
    currentSession, 
    currentPlan,
    loadSessions, 
    selectSession,
    isPlanning,
  } = useChatStore();

  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  useEffect(() => {
    if (sessionId) {
      selectSession(parseInt(sessionId, 10));
    }
  }, [sessionId, selectSession]);

  return (
    <div className="h-full flex">
      {/* Sidebar - Session List */}
      <div className="w-64 border-r border-gray-200 bg-white flex flex-col">
        <SessionList />
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col bg-gray-50">
        {currentSession ? (
          <>
            {/* Chat Header */}
            <div className="bg-white border-b border-gray-200 px-4 py-3">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="font-semibold text-gray-900">
                    Session #{currentSession.id}
                  </h2>
                  <p className="text-sm text-gray-500">
                    {currentSession.task_type} - {currentSession.status}
                  </p>
                </div>
                {isPlanning && (
                  <div className="flex items-center gap-2 text-primary-600">
                    <div className="flex gap-1">
                      <span className="thinking-dot w-2 h-2 bg-primary-500 rounded-full" />
                      <span className="thinking-dot w-2 h-2 bg-primary-500 rounded-full" />
                      <span className="thinking-dot w-2 h-2 bg-primary-500 rounded-full" />
                    </div>
                    <span className="text-sm">Planning...</span>
                  </div>
                )}
              </div>
            </div>

            {/* Messages and Plan */}
            <div className="flex-1 flex overflow-hidden">
              {/* Messages */}
              <div className="flex-1 flex flex-col">
                <MessageList />
                <ChatInput />
              </div>

              {/* Plan Viewer */}
              {currentPlan && (
                <div className="w-96 border-l border-gray-200 bg-white overflow-auto">
                  <PlanViewer plan={currentPlan} />
                </div>
              )}
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-gray-500">
            <div className="text-center">
              <p className="text-lg mb-2">No session selected</p>
              <p className="text-sm">Create a new session or select one from the sidebar</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
