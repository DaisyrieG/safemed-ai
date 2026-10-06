import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import QueryLab from './pages/QueryLab'
import Results from './pages/Results'
import DataSheet from './pages/DataSheet'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          {/* Main Landing is SafeMed AI Verified Clinical Search Assistant */}
          <Route index element={<QueryLab />} />
          <Route path="lab" element={<QueryLab />} />
          <Route path="results" element={<Results />} />
          <Route path="sheet" element={<DataSheet />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
