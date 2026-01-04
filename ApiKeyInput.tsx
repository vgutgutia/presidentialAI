import { useState, useEffect } from 'react'
import { getApiKey, setApiKey, clearApiKey, testApiKey, hasDefaultKey } from '../services/roboflow'

interface ApiKeyInputProps {
  onKeySet: (hasKey: boolean) => void
}

export default function ApiKeyInput({ onKeySet }: ApiKeyInputProps) {
  const [key, setKey] = useState('')
  const [isValidating, setIsValidating] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [hasStoredKey, setHasStoredKey] = useState(false)
  const [isDefaultKey, setIsDefaultKey] = useState(false)
  const [showCustomInput, setShowCustomInput] = useState(false)

  useEffect(() => {
    const storedKey = getApiKey()
    if (storedKey) {
      setHasStoredKey(true)
      setIsDefaultKey(hasDefaultKey())
      onKeySet(true)
    }
  }, [onKeySet])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!key.trim()) {
      setError('Please enter an API key')
      return
    }

    setIsValidating(true)
    setError(null)

    const isValid = await testApiKey(key.trim())
    
    if (isValid) {
      setApiKey(key.trim())
      setHasStoredKey(true)
      setIsDefaultKey(false)
      setShowCustomInput(false)
      onKeySet(true)
    } else {
      setError('Invalid API key. Please check and try again.')
    }
    
    setIsValidating(false)
  }

  const handleClear = () => {
    clearApiKey()
    setKey('')
    setHasStoredKey(false)
    setIsDefaultKey(false)
    onKeySet(false)
  }

  const handleUseDefault = () => {
    setApiKey('YcC1FIz8R2xm9oyPlpnp')
    setHasStoredKey(true)
    setIsDefaultKey(true)
    setShowCustomInput(false)
    onKeySet(true)
  }

  if (hasStoredKey && !showCustomInput) {
    return (
      <div className="api-key-status">
        <div className="status-content">
          <span className="status-icon">🤖</span>
          <div className="status-info">
            <span className="status-text">Roboflow AI Connected</span>
            {isDefaultKey && (
              <span className="status-subtitle">Using demo API key</span>
            )}
          </div>
          <button 
            onClick={() => setShowCustomInput(true)}
            className="btn btn-ghost btn-sm"
            type="button"
          >
            Change
          </button>
        </div>
        
        <style>{`
          .api-key-status {
            background: var(--eco-green-dim);
            border: 1px solid var(--eco-green);
            border-radius: var(--radius-md);
            padding: var(--space-3) var(--space-4);
          }

          .status-content {
            display: flex;
            align-items: center;
            gap: var(--space-3);
          }

          .status-icon {
            font-size: 1.4rem;
          }

          .status-info {
            flex: 1;
            display: flex;
            flex-direction: column;
          }

          .status-text {
            color: var(--eco-green);
            font-weight: 500;
            font-size: var(--text-sm);
          }

          .status-subtitle {
            color: var(--text-muted);
            font-size: var(--text-xs);
          }
        `}</style>
      </div>
    )
  }

  return (
    <div className="api-key-input">
      <div className="api-key-header">
        <h4>🤖 AI Detection Setup</h4>
        <p className="text-muted">
          {hasStoredKey 
            ? 'Enter a custom Roboflow API key or use the demo key.'
            : 'Connect to Roboflow for real AI-powered waste detection.'
          }
        </p>
      </div>

      <form onSubmit={handleSubmit} className="api-key-form">
        <div className="input-group">
          <input
            type="password"
            value={key}
            onChange={(e) => setKey(e.target.value)}
            placeholder="Enter your Roboflow API key..."
            className="input"
            disabled={isValidating}
          />
          <button 
            type="submit" 
            className="btn btn-primary"
            disabled={isValidating || !key.trim()}
          >
            {isValidating ? 'Validating...' : 'Connect'}
          </button>
        </div>
        
        {error && (
          <p className="error-message">{error}</p>
        )}
      </form>

      <div className="api-key-actions">
        <button 
          type="button"
          onClick={handleUseDefault}
          className="btn btn-secondary"
        >
          Use Demo API Key
        </button>
        {showCustomInput && (
          <button 
            type="button"
            onClick={() => setShowCustomInput(false)}
            className="btn btn-ghost"
          >
            Cancel
          </button>
        )}
      </div>

      <div className="api-key-help">
        <p className="text-muted">
          <strong>Get your own API key:</strong> <a href="https://roboflow.com" target="_blank" rel="noopener noreferrer">roboflow.com</a> → Settings → API Keys
        </p>
      </div>

      <style>{`
        .api-key-input {
          background: var(--bg-secondary);
          border: 1px solid var(--border-subtle);
          border-radius: var(--radius-lg);
          padding: var(--space-5);
        }

        .api-key-header {
          margin-bottom: var(--space-4);
        }

        .api-key-header h4 {
          margin-bottom: var(--space-2);
          font-size: var(--text-lg);
        }

        .api-key-form {
          margin-bottom: var(--space-4);
        }

        .input-group {
          display: flex;
          gap: var(--space-3);
        }

        .input-group .input {
          flex: 1;
        }

        .error-message {
          color: var(--status-danger);
          font-size: var(--text-sm);
          margin-top: var(--space-2);
        }

        .api-key-actions {
          display: flex;
          gap: var(--space-3);
          margin-bottom: var(--space-4);
        }

        .api-key-help {
          background: var(--bg-tertiary);
          border-radius: var(--radius-md);
          padding: var(--space-3);
          font-size: var(--text-sm);
        }

        .api-key-help a {
          color: var(--accent-primary);
        }
      `}</style>
    </div>
  )
}
