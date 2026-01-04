interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg'
  text?: string
}

export default function LoadingSpinner({ size = 'md', text }: LoadingSpinnerProps) {
  const sizeMap = {
    sm: '24px',
    md: '40px',
    lg: '64px'
  }

  return (
    <div className="flex flex-col items-center justify-center gap-4">
      <div 
        className="animate-pulse-glow"
        style={{
          width: sizeMap[size],
          height: sizeMap[size],
          borderRadius: '50%',
          border: '3px solid var(--accent-primary-dim)',
          borderTopColor: 'var(--accent-primary)',
          animation: 'spin 1s linear infinite, pulse-glow 2s ease-in-out infinite'
        }}
      />
      {text && <p className="text-muted">{text}</p>}
      
      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  )
}


