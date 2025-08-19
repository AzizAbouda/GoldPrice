import ChartCard from '../components/ChartCard'

function About() {
  return (
    <div className="max-w-3xl">
      <ChartCard title="About" subtitle="Gold Price Estimator UI">
        <p className="text-sm text-white/80 leading-6">
          This interface visualizes historical gold prices and model forecasts served by the backend API.
          Charts are animated with Framer Motion and rendered using Recharts. The layout is responsive and
          uses a modern card-based design.
        </p>
      </ChartCard>
    </div>
  )
}

export default About


