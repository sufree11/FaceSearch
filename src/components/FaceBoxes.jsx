import { useState } from 'react'

// An image with the recognised faces outlined and labelled.
export default function FaceBoxes({ src, faces = [], width, height, alt = '', className = '' }) {
  const [natural, setNatural] = useState(null)
  const W = width || natural?.[0]
  const H = height || natural?.[1]

  return (
    <div className={`face-boxes ${className}`}>
      <img
        src={src}
        alt={alt}
        onLoad={(e) => setNatural([e.target.naturalWidth, e.target.naturalHeight])}
      />
      {W &&
        H &&
        faces.map((f, i) => {
          const [x, y, w, h] = f.box
          return (
            <div
              key={i}
              className={`box ${f.name ? 'known' : 'unknown'}`}
              style={{
                left: `${(x / W) * 100}%`,
                top: `${(y / H) * 100}%`,
                width: `${(w / W) * 100}%`,
                height: `${(h / H) * 100}%`,
              }}
            >
              <span className="box-label">
                {f.name || 'Unknown'}
                {f.score ? ` · ${f.score.toFixed(2)}` : ''}
              </span>
            </div>
          )
        })}
    </div>
  )
}
