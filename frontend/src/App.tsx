import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import Workspace from './pages/Workspace'
import QueryLab from './pages/QueryLab'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Workspace />} />
          <Route path="classic" element={<div className="h-full overflow-y-auto"><QueryLab /></div>} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
