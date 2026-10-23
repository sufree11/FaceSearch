function peopleIn(image) {
  return [...new Set(image.faces.filter((f) => f.name).map((f) => f.name))]
}

export default function ImageCard({ image, index, onOpen }) {
  const people = peopleIn(image)
  const unknown = image.faces.filter((f) => !f.name).length

  return (
    <button type="button" className="image-card" onClick={onOpen}>
      <div className="image-card-media">
        <img src={image.url} alt={image.description || image.title || ''} loading="lazy" />
        {index != null && <span className="badge">{index}</span>}
        {image.match === 'source' && (
          <span className="flag" title="Labelled by the source; face not confirmed">
            face not confirmed
          </span>
        )}
      </div>
      <div className="image-card-body">
        <div className="chips">
          {people.map((p) => (
            <span key={p} className="chip">
              {p}
            </span>
          ))}
          {unknown > 0 && <span className="chip muted">+{unknown} unknown</span>}
        </div>
        <div className="date">{image.date || 'Date unknown'}</div>
        {image.description && <p className="desc">{image.description}</p>}
      </div>
    </button>
  )
}
