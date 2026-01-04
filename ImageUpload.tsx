import { useState, useCallback } from 'react'

interface ImageUploadProps {
  onUpload: (file: File) => void
  isLoading?: boolean
}

export default function ImageUpload({ onUpload, isLoading = false }: ImageUploadProps) {
  const [isDragActive, setIsDragActive] = useState(false)

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setIsDragActive(true)
    } else if (e.type === 'dragleave') {
      setIsDragActive(false)
    }
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragActive(false)
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onUpload(e.dataTransfer.files[0])
    }
  }, [onUpload])

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onUpload(e.target.files[0])
    }
  }

  return (
    <div
      className={`upload-zone ${isDragActive ? 'active' : ''}`}
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
    >
      <input
        type="file"
        id="file-upload"
        accept="image/*"
        onChange={handleChange}
        style={{ display: 'none' }}
        disabled={isLoading}
      />
      <label htmlFor="file-upload" style={{ cursor: isLoading ? 'wait' : 'pointer' }}>
        <div className="upload-icon">
          {isLoading ? '⏳' : '📤'}
        </div>
        <h4>{isLoading ? 'Analyzing...' : 'Upload Satellite Image'}</h4>
        <p className="text-muted">
          {isLoading 
            ? 'Our AI is scanning for waste accumulation zones...'
            : 'Drag & drop an image or click to browse'
          }
        </p>
      </label>
    </div>
  )
}


