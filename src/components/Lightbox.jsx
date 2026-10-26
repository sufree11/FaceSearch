import { useEffect } from 'react'
import FaceBoxes from './FaceBoxes.jsx'

const DATE_SOURCE = {
  exif: 'from the photo file (EXIF)',
  source: 'from the source page',
  none: '',
}

export default function Lightbox({ images, index, onClose, onMove }) {
  const image = images[index]

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape') onClose()
      if (e.key === 'ArrowRight') onMove(Math.min(index + 1, images.length - 1))
      if (e.key === 'ArrowLeft') onMove(Math.max(index - 1, 0))
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [index, images.length, onClose, onMove])

  if (!image) return null
  const rows = [
    ['Date', image.date ? `${image.date} ${DATE_SOURCE[image.date_source] ? `(${DATE_SOURCE[image.date_source]})` : ''}` : 'Unknown'],
    [
      'People',
      image.faces.length
        ? image.faces.map((f) => `${f.name || 'Unknown'}${f.name ? ` (${f.score.toFixed(2)})` : ''}`).join(', ')
        : 'No faces detected',
    ],
    ['Title', image.title],
    ['Author', image.author],
    ['License', image.license],
    ['Camera', image.camera],
    ['GPS', image.gps ? `${image.gps.lat}, ${image.gps.lon}` : null],
    ['Size', image.width ? `${image.width} × ${image.height}px` : null],
    ['File', image.url.replace('/media/', '')],
  ].filter(([, v]) => v)

  return (
    <div className="lightbox" role="dialog" aria-modal="true" onClick={onClose}>
      <div className="lightbox-inner" onClick={(e) => e.stopPropagation()}>
        <div className="lightbox-media">
          <FaceBoxes src={image.url} faces={image.faces} width={image.width} height={image.height} />
          <div className="lightbox-nav">
            <button type="button" disabled={index === 0} onClick={() => onMove(index - 1)}>
              ← Prev
            </button>
            <span>
              {index + 1} / {images.length}
            </span>
            <button type="button" disabled={index === images.length - 1} onClick={() => onMove(index + 1)}>
              Next →
            </button>
          </div>
        </div>
        <aside className="lightbox-info">
          <button type="button" className="close" onClick={onClose} aria-label="Close">
            ×
          </button>
          <h2>{image.desc_source === 'caption' ? 'Description (from Wikimedia Commons)' : 'AI description'}</h2>
          {image.desc_source === 'caption' && (
            <p className="pending-note">The AI description is being written in the background; search again later to see it.</p>
          )}
          <p>{image.description || 'No description yet.'}</p>
          {image.original_caption && image.original_caption !== image.description && (
            <>
              <h3>Source caption</h3>
              <p className="muted-text">{image.original_caption}</p>
            </>
          )}
          <h3>Metadata</h3>
          <dl>
            {rows.map(([k, v]) => (
              <div key={k}>
                <dt>{k}</dt>
                <dd>{v}</dd>
              </div>
            ))}
          </dl>
          <div className="lightbox-links">
            <a href={image.url} target="_blank" rel="noreferrer">
              Open full image
            </a>
            {image.source_url && (
              <a href={image.source_url} target="_blank" rel="noreferrer">
                Source page ↗
              </a>
            )}
          </div>
        </aside>
      </div>
    </div>
  )
}
