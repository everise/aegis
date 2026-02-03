/**
 * Main App component with routing.
 */

import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/common/Layout';
import ChatPage from './pages/ChatPage';
import TrainingPage from './pages/TrainingPage';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigate to="/chat" replace />} />
          <Route path="chat" element={<ChatPage />} />
          <Route path="chat/:sessionId" element={<ChatPage />} />
          <Route path="training" element={<TrainingPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
