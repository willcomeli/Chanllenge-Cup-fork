import { lazy, Suspense, useEffect } from 'react'
import AppLayout from './components/layout/AppLayout'
import DashboardPage from './pages/DashboardPage'
import EvolutionPage from './pages/EvolutionPage'
import MatchPage from './pages/MatchPage'
import ResumePage from './pages/ResumePage'
import ReviewPage from './pages/ReviewPage'
import { useAppRouter } from './router'

const GraphPage = lazy(() => import('./pages/GraphPage'))

function RedirectToDashboard() {
  const { navigate } = useAppRouter()

  useEffect(() => {
    navigate('/dashboard')
  }, [navigate])

  return null
}

export default function App() {
  const { path } = useAppRouter()
  const pages: Record<string, React.ReactNode> = {
    '/dashboard': <DashboardPage />,
    '/graph': <Suspense fallback={<div className="graph-loading"><span /><strong>正在加载三维图谱</strong><small>初始化 WebGL 场景...</small></div>}><GraphPage /></Suspense>,
    '/evolution': <EvolutionPage />,
    '/resume': <ResumePage />,
    '/match': <MatchPage />,
    '/review': <ReviewPage />,
  }

  return <AppLayout>{pages[path] ?? <RedirectToDashboard />}</AppLayout>
}
