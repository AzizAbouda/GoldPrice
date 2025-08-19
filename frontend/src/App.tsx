import { Routes, Route, Navigate } from 'react-router-dom'
import { AnimatePresence } from 'framer-motion'
import AppLayout from './layouts/AppLayout'
import Dashboard from './pages/Dashboard'
import Predictions from './pages/Predictions.tsx'
import Calculations from './pages/Calculations.tsx'
import About from './pages/About.tsx'
import './App.css'

function App() {
  return (
    <AnimatePresence mode="wait">
      <Routes>
        <Route path="/" element={<AppLayout />}>
          <Route index element={<Dashboard />} />
          <Route path="predictions" element={<Predictions />} />
          <Route path="calculations" element={<Calculations />} />
          <Route path="about" element={<About />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </AnimatePresence>
  )
}

export default App
