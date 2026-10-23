// Shows just the face from a photo, using the face box from the backend.
export default function FaceThumb({ src, box, imageSize, size = 40, alt = '' }) {
  if (!src) return <span className="face-thumb empty" style={{ width: size, height: size }} />
  if (!box || !imageSize?.[0]) {
    return <img className="face-thumb" src={src} alt={alt} width={size} height={size} />
  }
  const [x, y, w, h] = box
  const [W, H] = imageSize
  const crop = Math.max(w, h) * 1.5
  const scale = size / crop
  const cx = x + w / 2
  const cy = y + h / 2
  return (
    <span
      className="face-thumb"
      role="img"
      aria-label={alt}
      style={{
        width: size,
        height: size,
        backgroundImage: `url("${src}")`,
        backgroundSize: `${W * scale}px ${H * scale}px`,
        backgroundPosition: `${-(cx - crop / 2) * scale}px ${-(cy - crop / 2) * scale}px`,
      }}
    />
  )
}
